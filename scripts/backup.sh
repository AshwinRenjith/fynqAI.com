#!/bin/bash
# fynqAI Backup and Disaster Recovery Script
set -euo pipefail

# Configuration
NAMESPACE="fynqai"
BACKUP_BUCKET="fynqai-backups"
BACKUP_RETENTION_DAYS=30
BACKUP_DIR="/tmp/fynqai-backups"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace
    if ! kubectl get namespace ${NAMESPACE} &> /dev/null; then
        log_error "Namespace ${NAMESPACE} does not exist"
        exit 1
    fi
    
    # Create backup directory
    mkdir -p ${BACKUP_DIR}
    
    log_success "Prerequisites check passed"
}

# Database backup
backup_database() {
    local backup_name="database-$(date +%Y%m%d-%H%M%S).sql"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    
    log_info "Creating database backup: ${backup_name}"
    
    # Get postgres pod
    local postgres_pod=$(kubectl get pods -n ${NAMESPACE} -l app=fynqai-postgres -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$postgres_pod" ]; then
        log_error "PostgreSQL pod not found"
        return 1
    fi
    
    # Create database dump
    if kubectl exec -n ${NAMESPACE} ${postgres_pod} -- pg_dump -U fynqai fynqai > ${backup_path}; then
        log_success "Database backup created: ${backup_path}"
        
        # Compress backup
        gzip ${backup_path}
        log_success "Database backup compressed: ${backup_path}.gz"
        
        return 0
    else
        log_error "Database backup failed"
        return 1
    fi
}

# Redis backup
backup_redis() {
    local backup_name="redis-$(date +%Y%m%d-%H%M%S).rdb"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    
    log_info "Creating Redis backup: ${backup_name}"
    
    # Get redis pod
    local redis_pod=$(kubectl get pods -n ${NAMESPACE} -l app=fynqai-redis -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$redis_pod" ]; then
        log_error "Redis pod not found"
        return 1
    fi
    
    # Create Redis backup
    if kubectl exec -n ${NAMESPACE} ${redis_pod} -- redis-cli BGSAVE; then
        # Wait for backup to complete
        sleep 5
        
        # Copy RDB file
        kubectl cp ${NAMESPACE}/${redis_pod}:/data/dump.rdb ${backup_path}
        
        # Compress backup
        gzip ${backup_path}
        log_success "Redis backup created: ${backup_path}.gz"
        
        return 0
    else
        log_error "Redis backup failed"
        return 1
    fi
}

# Kubernetes configuration backup
backup_k8s_configs() {
    local backup_name="k8s-configs-$(date +%Y%m%d-%H%M%S)"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    
    log_info "Creating Kubernetes configuration backup: ${backup_name}"
    
    mkdir -p ${backup_path}
    
    # Backup all resources in namespace
    local resources=(
        "deployments"
        "services"
        "configmaps"
        "secrets"
        "persistentvolumeclaims"
        "ingresses"
        "horizontalpodautoscalers"
        "networkpolicies"
        "servicemonitors"
        "prometheusrules"
    )
    
    for resource in "${resources[@]}"; do
        log_info "Backing up ${resource}..."
        kubectl get ${resource} -n ${NAMESPACE} -o yaml > ${backup_path}/${resource}.yaml 2>/dev/null || true
    done
    
    # Backup cluster-wide resources
    kubectl get clusterroles -l app.kubernetes.io/part-of=fynqai -o yaml > ${backup_path}/clusterroles.yaml 2>/dev/null || true
    kubectl get clusterrolebindings -l app.kubernetes.io/part-of=fynqai -o yaml > ${backup_path}/clusterrolebindings.yaml 2>/dev/null || true
    
    # Create archive
    tar -czf ${backup_path}.tar.gz -C ${BACKUP_DIR} ${backup_name}
    rm -rf ${backup_path}
    
    log_success "Kubernetes configuration backup created: ${backup_path}.tar.gz"
}

# Application data backup
backup_app_data() {
    local backup_name="app-data-$(date +%Y%m%d-%H%M%S)"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    
    log_info "Creating application data backup: ${backup_name}"
    
    mkdir -p ${backup_path}
    
    # Backup persistent volumes
    local pvcs=$(kubectl get pvc -n ${NAMESPACE} -o jsonpath='{.items[*].metadata.name}')
    
    for pvc in $pvcs; do
        log_info "Backing up PVC: ${pvc}"
        
        # Create temporary pod to access PVC
        local temp_pod="backup-${pvc}-$(date +%s)"
        
        kubectl run ${temp_pod} -n ${NAMESPACE} --image=busybox --rm --restart=Never \
            --overrides="{
                \"spec\": {
                    \"containers\": [{
                        \"name\": \"backup\",
                        \"image\": \"busybox\",
                        \"command\": [\"tar\", \"czf\", \"/backup/${pvc}.tar.gz\", \"-C\", \"/data\", \".\"],
                        \"volumeMounts\": [{
                            \"name\": \"data\",
                            \"mountPath\": \"/data\"
                        }, {
                            \"name\": \"backup\",
                            \"mountPath\": \"/backup\"
                        }]
                    }],
                    \"volumes\": [{
                        \"name\": \"data\",
                        \"persistentVolumeClaim\": {
                            \"claimName\": \"${pvc}\"
                        }
                    }, {
                        \"name\": \"backup\",
                        \"hostPath\": {
                            \"path\": \"${backup_path}\"
                        }
                    }]
                }
            }" || log_warning "Failed to backup PVC: ${pvc}"
    done
    
    # Create archive
    if [ -n "$(ls -A ${backup_path} 2>/dev/null)" ]; then
        tar -czf ${backup_path}.tar.gz -C ${BACKUP_DIR} ${backup_name}
        rm -rf ${backup_path}
        log_success "Application data backup created: ${backup_path}.tar.gz"
    else
        log_warning "No application data to backup"
        rm -rf ${backup_path}
    fi
}

# Full backup
full_backup() {
    log_info "=== Starting Full Backup ==="
    
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local full_backup_dir="${BACKUP_DIR}/full-backup-${timestamp}"
    
    mkdir -p ${full_backup_dir}
    
    # Backup database
    backup_database
    
    # Backup Redis
    backup_redis
    
    # Backup Kubernetes configurations
    backup_k8s_configs
    
    # Backup application data
    backup_app_data
    
    # Move all backups to full backup directory
    mv ${BACKUP_DIR}/*.gz ${full_backup_dir}/ 2>/dev/null || true
    mv ${BACKUP_DIR}/*.tar.gz ${full_backup_dir}/ 2>/dev/null || true
    
    # Create backup manifest
    cat > ${full_backup_dir}/manifest.txt << EOF
fynqAI Full Backup
Timestamp: ${timestamp}
Namespace: ${NAMESPACE}
Backup Contents:
$(ls -la ${full_backup_dir})

Kubernetes Version: $(kubectl version --short --client)
Cluster Info: $(kubectl cluster-info | head -1)
EOF
    
    # Create final archive
    tar -czf ${BACKUP_DIR}/fynqai-full-backup-${timestamp}.tar.gz -C ${BACKUP_DIR} full-backup-${timestamp}
    rm -rf ${full_backup_dir}
    
    log_success "Full backup completed: ${BACKUP_DIR}/fynqai-full-backup-${timestamp}.tar.gz"
}

# Restore database
restore_database() {
    local backup_file=$1
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    log_info "Restoring database from: $backup_file"
    
    # Get postgres pod
    local postgres_pod=$(kubectl get pods -n ${NAMESPACE} -l app=fynqai-postgres -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$postgres_pod" ]; then
        log_error "PostgreSQL pod not found"
        return 1
    fi
    
    # Confirm restore
    read -p "Are you sure you want to restore the database? This will overwrite all current data. (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        log_warning "Database restore cancelled"
        return 1
    fi
    
    # Stop API services
    log_info "Scaling down API services..."
    kubectl scale deployment fynqai-api -n ${NAMESPACE} --replicas=0
    kubectl scale deployment fynqai-worker -n ${NAMESPACE} --replicas=0
    kubectl scale deployment fynqai-scheduler -n ${NAMESPACE} --replicas=0
    
    # Wait for pods to terminate
    sleep 10
    
    # Decompress if needed
    local restore_file=$backup_file
    if [[ $backup_file == *.gz ]]; then
        restore_file="${backup_file%.gz}"
        gunzip -c $backup_file > $restore_file
    fi
    
    # Restore database
    if kubectl exec -n ${NAMESPACE} ${postgres_pod} -- psql -U fynqai -d fynqai < $restore_file; then
        log_success "Database restored successfully"
        
        # Scale services back up
        log_info "Scaling services back up..."
        kubectl scale deployment fynqai-api -n ${NAMESPACE} --replicas=2
        kubectl scale deployment fynqai-worker -n ${NAMESPACE} --replicas=3
        kubectl scale deployment fynqai-scheduler -n ${NAMESPACE} --replicas=1
        
        # Clean up temporary file
        if [[ $backup_file == *.gz ]]; then
            rm -f $restore_file
        fi
        
        return 0
    else
        log_error "Database restore failed"
        
        # Scale services back up anyway
        kubectl scale deployment fynqai-api -n ${NAMESPACE} --replicas=2
        kubectl scale deployment fynqai-worker -n ${NAMESPACE} --replicas=3
        kubectl scale deployment fynqai-scheduler -n ${NAMESPACE} --replicas=1
        
        return 1
    fi
}

# Restore Redis
restore_redis() {
    local backup_file=$1
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    log_info "Restoring Redis from: $backup_file"
    
    # Get redis pod
    local redis_pod=$(kubectl get pods -n ${NAMESPACE} -l app=fynqai-redis -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$redis_pod" ]; then
        log_error "Redis pod not found"
        return 1
    fi
    
    # Confirm restore
    read -p "Are you sure you want to restore Redis? This will overwrite all current cache data. (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        log_warning "Redis restore cancelled"
        return 1
    fi
    
    # Decompress if needed
    local restore_file=$backup_file
    if [[ $backup_file == *.gz ]]; then
        restore_file="${backup_file%.gz}"
        gunzip -c $backup_file > $restore_file
    fi
    
    # Stop Redis
    kubectl exec -n ${NAMESPACE} ${redis_pod} -- redis-cli SHUTDOWN SAVE || true
    
    # Wait for Redis to stop
    sleep 5
    
    # Copy RDB file
    kubectl cp $restore_file ${NAMESPACE}/${redis_pod}:/data/dump.rdb
    
    # Restart Redis pod
    kubectl delete pod ${redis_pod} -n ${NAMESPACE}
    
    # Wait for pod to restart
    kubectl wait --for=condition=ready pod -l app=fynqai-redis -n ${NAMESPACE} --timeout=300s
    
    log_success "Redis restored successfully"
    
    # Clean up temporary file
    if [[ $backup_file == *.gz ]]; then
        rm -f $restore_file
    fi
}

# Disaster recovery
disaster_recovery() {
    local backup_archive=$1
    
    if [ ! -f "$backup_archive" ]; then
        log_error "Backup archive not found: $backup_archive"
        return 1
    fi
    
    log_info "=== Starting Disaster Recovery ==="
    log_warning "This will completely restore the fynqAI application from backup"
    
    read -p "Are you sure you want to proceed with disaster recovery? (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        log_warning "Disaster recovery cancelled"
        return 1
    fi
    
    # Extract backup
    local recovery_dir="${BACKUP_DIR}/recovery-$(date +%s)"
    mkdir -p ${recovery_dir}
    
    log_info "Extracting backup archive..."
    tar -xzf $backup_archive -C ${recovery_dir}
    
    # Find backup components
    local backup_contents=$(find ${recovery_dir} -name "*.gz" -o -name "*.tar.gz")
    
    # Delete existing resources
    log_info "Removing existing resources..."
    kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
    
    # Wait for namespace deletion
    while kubectl get namespace ${NAMESPACE} &> /dev/null; do
        log_info "Waiting for namespace deletion..."
        sleep 5
    done
    
    # Recreate namespace
    kubectl create namespace ${NAMESPACE}
    
    # Restore Kubernetes configurations
    local k8s_backup=$(find ${recovery_dir} -name "k8s-configs-*.tar.gz" | head -1)
    if [ -n "$k8s_backup" ]; then
        log_info "Restoring Kubernetes configurations..."
        local k8s_dir="${recovery_dir}/k8s-configs"
        mkdir -p ${k8s_dir}
        tar -xzf $k8s_backup -C ${k8s_dir}
        
        # Apply configurations in order
        kubectl apply -f ${k8s_dir}/*/secrets.yaml
        kubectl apply -f ${k8s_dir}/*/configmaps.yaml
        kubectl apply -f ${k8s_dir}/*/persistentvolumeclaims.yaml
        kubectl apply -f ${k8s_dir}/*/deployments.yaml
        kubectl apply -f ${k8s_dir}/*/services.yaml
        kubectl apply -f ${k8s_dir}/*/ingresses.yaml
        kubectl apply -f ${k8s_dir}/*/horizontalpodautoscalers.yaml
        kubectl apply -f ${k8s_dir}/*/networkpolicies.yaml
        kubectl apply -f ${k8s_dir}/*/servicemonitors.yaml
        kubectl apply -f ${k8s_dir}/*/prometheusrules.yaml
    fi
    
    # Wait for pods to be ready
    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=fynqai-postgres -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=ready pod -l app=fynqai-redis -n ${NAMESPACE} --timeout=300s
    
    # Restore database
    local db_backup=$(find ${recovery_dir} -name "database-*.sql.gz" | head -1)
    if [ -n "$db_backup" ]; then
        restore_database $db_backup
    fi
    
    # Restore Redis
    local redis_backup=$(find ${recovery_dir} -name "redis-*.rdb.gz" | head -1)
    if [ -n "$redis_backup" ]; then
        restore_redis $redis_backup
    fi
    
    # Clean up recovery directory
    rm -rf ${recovery_dir}
    
    log_success "Disaster recovery completed successfully"
}

# Clean old backups
cleanup_backups() {
    log_info "Cleaning up old backups..."
    
    # Remove backups older than retention period
    find ${BACKUP_DIR} -name "*.gz" -o -name "*.tar.gz" -mtime +${BACKUP_RETENTION_DAYS} -delete
    
    log_success "Old backups cleaned up (retention: ${BACKUP_RETENTION_DAYS} days)"
}

# Upload to cloud storage (S3 compatible)
upload_backup() {
    local backup_file=$1
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Check if AWS CLI is available
    if command -v aws &> /dev/null; then
        log_info "Uploading backup to S3: ${BACKUP_BUCKET}"
        
        local filename=$(basename $backup_file)
        if aws s3 cp $backup_file s3://${BACKUP_BUCKET}/$filename; then
            log_success "Backup uploaded to S3: s3://${BACKUP_BUCKET}/$filename"
            return 0
        else
            log_error "Failed to upload backup to S3"
            return 1
        fi
    else
        log_warning "AWS CLI not available, skipping cloud upload"
        return 1
    fi
}

# List backups
list_backups() {
    log_info "=== Local Backups ==="
    
    if [ -d "$BACKUP_DIR" ]; then
        ls -la ${BACKUP_DIR}/*.gz ${BACKUP_DIR}/*.tar.gz 2>/dev/null || echo "No local backups found"
    else
        echo "Backup directory does not exist"
    fi
    
    echo ""
    log_info "=== Cloud Backups (S3) ==="
    
    if command -v aws &> /dev/null; then
        aws s3 ls s3://${BACKUP_BUCKET}/ 2>/dev/null || echo "Cannot access S3 bucket or no cloud backups found"
    else
        echo "AWS CLI not available"
    fi
}

# Main function
main() {
    case ${1:-""} in
        "backup-db")
            check_prerequisites
            backup_database
            ;;
        "backup-redis")
            check_prerequisites
            backup_redis
            ;;
        "backup-k8s")
            check_prerequisites
            backup_k8s_configs
            ;;
        "backup-data")
            check_prerequisites
            backup_app_data
            ;;
        "backup-full")
            check_prerequisites
            full_backup
            ;;
        "restore-db")
            check_prerequisites
            restore_database $2
            ;;
        "restore-redis")
            check_prerequisites
            restore_redis $2
            ;;
        "disaster-recovery")
            check_prerequisites
            disaster_recovery $2
            ;;
        "cleanup")
            cleanup_backups
            ;;
        "upload")
            upload_backup $2
            ;;
        "list")
            list_backups
            ;;
        *)
            echo "fynqAI Backup and Disaster Recovery"
            echo ""
            echo "Usage: $0 <command> [args...]"
            echo ""
            echo "Backup Commands:"
            echo "  backup-db              Backup PostgreSQL database"
            echo "  backup-redis           Backup Redis data"
            echo "  backup-k8s             Backup Kubernetes configurations"
            echo "  backup-data            Backup application data (PVCs)"
            echo "  backup-full            Full backup (all components)"
            echo ""
            echo "Restore Commands:"
            echo "  restore-db <file>      Restore PostgreSQL database"
            echo "  restore-redis <file>   Restore Redis data"
            echo "  disaster-recovery <archive>  Complete disaster recovery"
            echo ""
            echo "Management Commands:"
            echo "  cleanup                Remove old backups"
            echo "  upload <file>          Upload backup to cloud storage"
            echo "  list                   List available backups"
            echo ""
            echo "Examples:"
            echo "  $0 backup-full"
            echo "  $0 restore-db /tmp/fynqai-backups/database-20240101-120000.sql.gz"
            echo "  $0 disaster-recovery /tmp/fynqai-backups/fynqai-full-backup-20240101-120000.tar.gz"
            exit 1
            ;;
    esac
}

main "$@"
