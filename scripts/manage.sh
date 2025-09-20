#!/bin/bash
# fynqAI Infrastructure Monitoring and Management Script
set -euo pipefail

# Configuration
NAMESPACE="fynqai"
MONITORING_NAMESPACE="monitoring"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get cluster status
cluster_status() {
    log_info "=== fynqAI Cluster Status ==="
    
    echo -e "\n${BLUE}Namespace Status:${NC}"
    kubectl get namespaces | grep -E "(fynqai|monitoring)" || echo "Namespaces not found"
    
    echo -e "\n${BLUE}Pod Status:${NC}"
    kubectl get pods -n ${NAMESPACE} -o wide 2>/dev/null || echo "No pods found in ${NAMESPACE}"
    
    echo -e "\n${BLUE}Service Status:${NC}"
    kubectl get services -n ${NAMESPACE} 2>/dev/null || echo "No services found in ${NAMESPACE}"
    
    echo -e "\n${BLUE}Ingress Status:${NC}"
    kubectl get ingress -n ${NAMESPACE} 2>/dev/null || echo "No ingress found in ${NAMESPACE}"
    
    echo -e "\n${BLUE}HPA Status:${NC}"
    kubectl get hpa -n ${NAMESPACE} 2>/dev/null || echo "No HPAs found in ${NAMESPACE}"
    
    echo -e "\n${BLUE}PVC Status:${NC}"
    kubectl get pvc -n ${NAMESPACE} 2>/dev/null || echo "No PVCs found in ${NAMESPACE}"
}

# Get resource usage
resource_usage() {
    log_info "=== Resource Usage ==="
    
    echo -e "\n${BLUE}Node Resource Usage:${NC}"
    kubectl top nodes 2>/dev/null || echo "Metrics server not available"
    
    echo -e "\n${BLUE}Pod Resource Usage:${NC}"
    kubectl top pods -n ${NAMESPACE} 2>/dev/null || echo "Metrics server not available"
    
    echo -e "\n${BLUE}Resource Requests vs Limits:${NC}"
    kubectl describe nodes | grep -A 5 "Allocated resources:" || echo "Node info not available"
}

# Check pod health
pod_health() {
    log_info "=== Pod Health Check ==="
    
    # Check pod status
    PODS=$(kubectl get pods -n ${NAMESPACE} -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
    
    if [ -z "$PODS" ]; then
        log_warning "No pods found in namespace ${NAMESPACE}"
        return
    fi
    
    for pod in $PODS; do
        echo -e "\n${BLUE}Pod: ${pod}${NC}"
        
        # Pod status
        STATUS=$(kubectl get pod ${pod} -n ${NAMESPACE} -o jsonpath='{.status.phase}')
        if [ "$STATUS" = "Running" ]; then
            echo -e "  Status: ${GREEN}${STATUS}${NC}"
        else
            echo -e "  Status: ${RED}${STATUS}${NC}"
        fi
        
        # Restart count
        RESTARTS=$(kubectl get pod ${pod} -n ${NAMESPACE} -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null || echo "0")
        if [ "$RESTARTS" -gt 0 ]; then
            echo -e "  Restarts: ${YELLOW}${RESTARTS}${NC}"
        else
            echo -e "  Restarts: ${GREEN}${RESTARTS}${NC}"
        fi
        
        # Ready status
        READY=$(kubectl get pod ${pod} -n ${NAMESPACE} -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
        if [ "$READY" = "true" ]; then
            echo -e "  Ready: ${GREEN}${READY}${NC}"
        else
            echo -e "  Ready: ${RED}${READY}${NC}"
        fi
        
        # Age
        AGE=$(kubectl get pod ${pod} -n ${NAMESPACE} -o jsonpath='{.metadata.creationTimestamp}' | xargs -I {} date -d {} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "Unknown")
        echo -e "  Age: ${AGE}"
    done
}

# Get application logs
app_logs() {
    local component=${1:-api}
    local lines=${2:-100}
    
    log_info "=== Application Logs (${component}) ==="
    
    case $component in
        api)
            SELECTOR="app=fynqai-api"
            ;;
        worker)
            SELECTOR="app=fynqai-worker"
            ;;
        scheduler)
            SELECTOR="app=fynqai-scheduler"
            ;;
        postgres)
            SELECTOR="app=fynqai-postgres"
            ;;
        redis)
            SELECTOR="app=fynqai-redis"
            ;;
        *)
            log_error "Unknown component: ${component}"
            echo "Available components: api, worker, scheduler, postgres, redis"
            return 1
            ;;
    esac
    
    kubectl logs -l ${SELECTOR} -n ${NAMESPACE} --tail=${lines} --timestamps
}

# Check API health endpoint
api_health() {
    log_info "=== API Health Check ==="
    
    # Get API service
    API_SERVICE=$(kubectl get service fynqai-api -n ${NAMESPACE} -o jsonpath='{.metadata.name}' 2>/dev/null || echo "")
    
    if [ -z "$API_SERVICE" ]; then
        log_error "API service not found"
        return 1
    fi
    
    # Port forward to API service
    log_info "Port forwarding to API service..."
    kubectl port-forward svc/fynqai-api -n ${NAMESPACE} 8080:80 &
    PF_PID=$!
    
    sleep 3
    
    # Test health endpoint
    if curl -f http://localhost:8080/health &>/dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
    fi
    
    # Cleanup
    kill $PF_PID &>/dev/null || true
}

# Scale services
scale_service() {
    local service=$1
    local replicas=$2
    
    log_info "Scaling ${service} to ${replicas} replicas..."
    
    case $service in
        api)
            kubectl scale deployment fynqai-api -n ${NAMESPACE} --replicas=${replicas}
            ;;
        worker)
            kubectl scale deployment fynqai-worker -n ${NAMESPACE} --replicas=${replicas}
            ;;
        *)
            log_error "Unknown service: ${service}"
            echo "Available services: api, worker"
            return 1
            ;;
    esac
    
    log_success "Scaled ${service} to ${replicas} replicas"
}

# Database operations
db_operations() {
    local operation=$1
    
    case $operation in
        backup)
            log_info "Creating database backup..."
            kubectl exec -n ${NAMESPACE} deployment/fynqai-postgres -- pg_dump -U fynqai fynqai > "backup-$(date +%Y%m%d-%H%M%S).sql"
            log_success "Database backup created"
            ;;
        migrate)
            log_info "Running database migrations..."
            kubectl exec -n ${NAMESPACE} deployment/fynqai-api -- alembic upgrade head
            log_success "Database migrations completed"
            ;;
        console)
            log_info "Opening database console..."
            kubectl exec -it -n ${NAMESPACE} deployment/fynqai-postgres -- psql -U fynqai -d fynqai
            ;;
        *)
            log_error "Unknown operation: ${operation}"
            echo "Available operations: backup, migrate, console"
            return 1
            ;;
    esac
}

# Monitoring dashboard
monitoring() {
    log_info "=== Monitoring Dashboard Access ==="
    
    # Check if monitoring namespace exists
    if ! kubectl get namespace ${MONITORING_NAMESPACE} &>/dev/null; then
        log_error "Monitoring namespace not found"
        return 1
    fi
    
    echo -e "\n${BLUE}Available Monitoring Services:${NC}"
    kubectl get services -n ${MONITORING_NAMESPACE}
    
    echo -e "\n${BLUE}Access Instructions:${NC}"
    echo "Grafana: kubectl port-forward svc/grafana -n ${MONITORING_NAMESPACE} 3000:3000"
    echo "Prometheus: kubectl port-forward svc/prometheus -n ${MONITORING_NAMESPACE} 9090:9090"
    echo "Flower (Celery): kubectl port-forward svc/flower -n ${NAMESPACE} 5555:5555"
    
    # Optionally start port forwarding
    read -p "Start Grafana port forwarding? (y/n): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Starting Grafana port forwarding on http://localhost:3000"
        kubectl port-forward svc/grafana -n ${MONITORING_NAMESPACE} 3000:3000
    fi
}

# Troubleshoot issues
troubleshoot() {
    log_info "=== Troubleshooting ==="
    
    echo -e "\n${BLUE}Recent Events:${NC}"
    kubectl get events -n ${NAMESPACE} --sort-by='.lastTimestamp' | tail -10
    
    echo -e "\n${BLUE}Failed Pods:${NC}"
    kubectl get pods -n ${NAMESPACE} --field-selector=status.phase=Failed
    
    echo -e "\n${BLUE}Pending Pods:${NC}"
    kubectl get pods -n ${NAMESPACE} --field-selector=status.phase=Pending
    
    echo -e "\n${BLUE}Pod Descriptions (Recent Issues):${NC}"
    FAILED_PODS=$(kubectl get pods -n ${NAMESPACE} --field-selector=status.phase=Failed -o jsonpath='{.items[*].metadata.name}')
    for pod in $FAILED_PODS; do
        echo -e "\n${YELLOW}Pod: ${pod}${NC}"
        kubectl describe pod ${pod} -n ${NAMESPACE} | tail -20
    done
    
    PENDING_PODS=$(kubectl get pods -n ${NAMESPACE} --field-selector=status.phase=Pending -o jsonpath='{.items[*].metadata.name}')
    for pod in $PENDING_PODS; do
        echo -e "\n${YELLOW}Pod: ${pod}${NC}"
        kubectl describe pod ${pod} -n ${NAMESPACE} | tail -20
    done
}

# Performance metrics
performance() {
    log_info "=== Performance Metrics ==="
    
    echo -e "\n${BLUE}Cluster Performance:${NC}"
    kubectl top nodes 2>/dev/null || echo "Metrics server not available"
    
    echo -e "\n${BLUE}Application Performance:${NC}"
    kubectl top pods -n ${NAMESPACE} 2>/dev/null || echo "Metrics server not available"
    
    echo -e "\n${BLUE}HPA Metrics:${NC}"
    kubectl get hpa -n ${NAMESPACE} -o wide 2>/dev/null || echo "No HPAs found"
    
    echo -e "\n${BLUE}Resource Quotas:${NC}"
    kubectl get resourcequota -n ${NAMESPACE} 2>/dev/null || echo "No resource quotas found"
    
    echo -e "\n${BLUE}Limit Ranges:${NC}"
    kubectl get limitrange -n ${NAMESPACE} 2>/dev/null || echo "No limit ranges found"
}

# Cleanup old resources
cleanup() {
    log_info "=== Cleanup Old Resources ==="
    
    # Clean up completed jobs
    log_info "Cleaning up completed jobs..."
    kubectl delete jobs -n ${NAMESPACE} --field-selector=status.successful=1
    
    # Clean up old replica sets
    log_info "Cleaning up old replica sets..."
    kubectl delete replicasets -n ${NAMESPACE} --field-selector=status.replicas=0
    
    # Clean up evicted pods
    log_info "Cleaning up evicted pods..."
    kubectl delete pods -n ${NAMESPACE} --field-selector=status.phase=Failed
    
    log_success "Cleanup completed"
}

# Main menu
show_menu() {
    echo -e "\n${BLUE}=== fynqAI Infrastructure Management ===${NC}"
    echo "1.  Cluster Status"
    echo "2.  Resource Usage"
    echo "3.  Pod Health"
    echo "4.  Application Logs"
    echo "5.  API Health Check"
    echo "6.  Scale Services"
    echo "7.  Database Operations"
    echo "8.  Monitoring Dashboard"
    echo "9.  Troubleshoot"
    echo "10. Performance Metrics"
    echo "11. Cleanup"
    echo "12. Exit"
    echo ""
}

# Main function
main() {
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option [1-12]: " choice
            case $choice in
                1) cluster_status ;;
                2) resource_usage ;;
                3) pod_health ;;
                4) 
                    read -p "Component [api/worker/scheduler/postgres/redis]: " comp
                    read -p "Lines [100]: " lines
                    lines=${lines:-100}
                    app_logs $comp $lines
                    ;;
                5) api_health ;;
                6) 
                    read -p "Service [api/worker]: " service
                    read -p "Replicas: " replicas
                    scale_service $service $replicas
                    ;;
                7) 
                    read -p "Operation [backup/migrate/console]: " op
                    db_operations $op
                    ;;
                8) monitoring ;;
                9) troubleshoot ;;
                10) performance ;;
                11) cleanup ;;
                12) exit 0 ;;
                *) log_error "Invalid option" ;;
            esac
            echo ""
            read -p "Press Enter to continue..."
        done
    else
        # Command line mode
        case $1 in
            status) cluster_status ;;
            resources) resource_usage ;;
            health) pod_health ;;
            logs) app_logs ${2:-api} ${3:-100} ;;
            api-health) api_health ;;
            scale) scale_service $2 $3 ;;
            db) db_operations $2 ;;
            monitor) monitoring ;;
            troubleshoot) troubleshoot ;;
            performance) performance ;;
            cleanup) cleanup ;;
            *)
                echo "Usage: $0 [command] [args...]"
                echo "Commands:"
                echo "  status              Show cluster status"
                echo "  resources           Show resource usage"
                echo "  health              Check pod health"
                echo "  logs [component] [lines]  Show application logs"
                echo "  api-health          Check API health endpoint"
                echo "  scale <service> <replicas>  Scale service"
                echo "  db <operation>      Database operations"
                echo "  monitor             Access monitoring dashboard"
                echo "  troubleshoot        Troubleshoot issues"
                echo "  performance         Show performance metrics"
                echo "  cleanup             Clean up old resources"
                exit 1
                ;;
        esac
    fi
}

main "$@"
