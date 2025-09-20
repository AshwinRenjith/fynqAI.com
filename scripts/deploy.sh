#!/bin/bash
# fynqAI Deployment Script - Production Kubernetes Deployment
set -euo pipefail

# Configuration
NAMESPACE="fynqai"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-fynqai}"
KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    
    # Check cluster access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Build image
    docker build -f docker/Dockerfile -t "${DOCKER_REGISTRY}/backend:${IMAGE_TAG}" .
    
    # Push image
    docker push "${DOCKER_REGISTRY}/backend:${IMAGE_TAG}"
    
    log_success "Docker image built and pushed: ${DOCKER_REGISTRY}/backend:${IMAGE_TAG}"
}

# Create namespace and secrets
setup_namespace() {
    log_info "Setting up namespace and secrets..."
    
    # Apply base manifests (namespace, secrets, configmaps)
    kubectl apply -f infrastructure/kubernetes/01-base.yaml
    
    # Wait for namespace to be ready
    kubectl wait --for=condition=Ready namespace/${NAMESPACE} --timeout=60s || true
    
    log_success "Namespace and base resources created"
}

# Deploy database and cache
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Apply database and cache deployments
    kubectl apply -f infrastructure/kubernetes/01-base.yaml
    
    # Wait for postgres to be ready
    log_info "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=Available deployment/fynqai-postgres -n ${NAMESPACE} --timeout=300s
    
    # Wait for redis to be ready
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=Available deployment/fynqai-redis -n ${NAMESPACE} --timeout=180s
    
    log_success "Infrastructure components deployed"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Create migration job
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: fynqai-migration-$(date +%s)
  namespace: ${NAMESPACE}
spec:
  template:
    spec:
      containers:
      - name: migration
        image: ${DOCKER_REGISTRY}/backend:${IMAGE_TAG}
        command: ["alembic", "upgrade", "head"]
        envFrom:
        - configMapRef:
            name: fynqai-config
        - secretRef:
            name: fynqai-secrets
        env:
        - name: DATABASE_URL
          value: "postgresql+asyncpg://fynqai:\$(DATABASE_PASSWORD)@\$(DATABASE_HOST):\$(DATABASE_PORT)/\$(DATABASE_NAME)"
      restartPolicy: Never
  backoffLimit: 3
EOF
    
    # Wait for migration to complete
    kubectl wait --for=condition=Complete job -l app=fynqai-migration -n ${NAMESPACE} --timeout=300s
    
    log_success "Database migrations completed"
}

# Deploy API services
deploy_api() {
    log_info "Deploying API services..."
    
    # Update image tag in manifests
    sed -i.bak "s|image: fynqai/backend:latest|image: ${DOCKER_REGISTRY}/backend:${IMAGE_TAG}|g" infrastructure/kubernetes/02-api.yaml
    
    # Apply API deployment
    kubectl apply -f infrastructure/kubernetes/02-api.yaml
    
    # Wait for API to be ready
    log_info "Waiting for API deployment to be ready..."
    kubectl wait --for=condition=Available deployment/fynqai-api -n ${NAMESPACE} --timeout=300s
    
    # Restore original manifest
    mv infrastructure/kubernetes/02-api.yaml.bak infrastructure/kubernetes/02-api.yaml
    
    log_success "API services deployed"
}

# Deploy worker services
deploy_workers() {
    log_info "Deploying worker services..."
    
    # Update image tag in manifests
    sed -i.bak "s|image: fynqai/backend:latest|image: ${DOCKER_REGISTRY}/backend:${IMAGE_TAG}|g" infrastructure/kubernetes/03-workers.yaml
    
    # Apply worker deployment
    kubectl apply -f infrastructure/kubernetes/03-workers.yaml
    
    # Wait for workers to be ready
    log_info "Waiting for worker deployment to be ready..."
    kubectl wait --for=condition=Available deployment/fynqai-worker -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=Available deployment/fynqai-scheduler -n ${NAMESPACE} --timeout=300s
    
    # Restore original manifest
    mv infrastructure/kubernetes/03-workers.yaml.bak infrastructure/kubernetes/03-workers.yaml
    
    log_success "Worker services deployed"
}

# Setup networking
setup_networking() {
    log_info "Setting up networking and ingress..."
    
    # Apply networking manifests
    kubectl apply -f infrastructure/kubernetes/04-networking.yaml
    
    log_success "Networking configured"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Apply monitoring manifests
    kubectl apply -f infrastructure/monitoring/prometheus.yaml
    
    log_success "Monitoring stack deployed"
}

# Health check
health_check() {
    log_info "Performing health checks..."
    
    # Wait for all pods to be ready
    kubectl wait --for=condition=Ready pod -l app=fynqai-api -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=Ready pod -l app=fynqai-worker -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=Ready pod -l app=fynqai-scheduler -n ${NAMESPACE} --timeout=300s
    
    # Check API health endpoint
    log_info "Checking API health endpoint..."
    API_URL=$(kubectl get ingress fynqai-ingress -n ${NAMESPACE} -o jsonpath='{.spec.rules[0].host}')
    
    if [ -n "${API_URL}" ]; then
        if curl -f "https://${API_URL}/health" &> /dev/null; then
            log_success "API health check passed"
        else
            log_warning "API health check failed (may need time to propagate)"
        fi
    else
        log_warning "Ingress not yet configured"
    fi
    
    # Display deployment status
    echo ""
    log_info "Deployment Status:"
    kubectl get pods -n ${NAMESPACE} -o wide
    echo ""
    kubectl get services -n ${NAMESPACE}
    echo ""
    kubectl get ingress -n ${NAMESPACE}
    
    log_success "Health checks completed"
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        log_error "Deployment failed. Check logs above for details."
        
        # Show recent events
        echo ""
        log_info "Recent cluster events:"
        kubectl get events -n ${NAMESPACE} --sort-by='.lastTimestamp' | tail -10
        
        exit 1
    fi
}

# Main deployment function
main() {
    log_info "Starting fynqAI production deployment..."
    log_info "Image: ${DOCKER_REGISTRY}/backend:${IMAGE_TAG}"
    log_info "Namespace: ${NAMESPACE}"
    log_info "Context: ${KUBECTL_CONTEXT}"
    
    # Set kubectl context
    kubectl config use-context ${KUBECTL_CONTEXT}
    
    # Run deployment steps
    check_prerequisites
    build_and_push_image
    setup_namespace
    deploy_infrastructure
    run_migrations
    deploy_api
    deploy_workers
    setup_networking
    deploy_monitoring
    health_check
    
    log_success "fynqAI deployment completed successfully!"
    log_info "API URL: https://$(kubectl get ingress fynqai-ingress -n ${NAMESPACE} -o jsonpath='{.spec.rules[0].host}')"
    log_info "Monitoring: kubectl port-forward svc/grafana -n monitoring 3000:3000"
}

# Set trap for cleanup
trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --context)
            KUBECTL_CONTEXT="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --image-tag TAG     Docker image tag (default: latest)"
            echo "  --registry REGISTRY Docker registry (default: fynqai)"
            echo "  --context CONTEXT   Kubectl context (default: production)"
            echo "  --skip-build        Skip Docker build and push"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Skip build if requested
if [ "${SKIP_BUILD:-false}" = "true" ]; then
    build_and_push_image() {
        log_info "Skipping Docker build as requested"
    }
fi

# Run main function
main
