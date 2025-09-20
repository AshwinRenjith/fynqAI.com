# fynqAI Production Deployment Guide

## 🚀 Production Readiness Checklist

### Prerequisites
- [ ] Kubernetes cluster (v1.24+) with sufficient resources
- [ ] kubectl configured with cluster admin access
- [ ] Docker registry access for image storage
- [ ] Domain name and SSL certificates
- [ ] Monitoring infrastructure (Prometheus/Grafana)
- [ ] Backup storage (S3-compatible)

### Infrastructure Components

#### ✅ 1. Containerization
- [x] Multi-stage Dockerfile with security hardening
- [x] Non-root user execution
- [x] Health check endpoints
- [x] Minimal attack surface

#### ✅ 2. Development Environment
- [x] Docker Compose for local development
- [x] All services (API, Workers, PostgreSQL, Redis)
- [x] Volume mounts for development
- [x] Environment variable configuration

#### ✅ 3. Production Environment
- [x] Production Docker Compose with HA
- [x] Service replicas and scaling
- [x] Resource limits and requests
- [x] Health checks and restart policies

#### ✅ 4. Kubernetes Orchestration
- [x] Base infrastructure (Namespace, ConfigMaps, Secrets)
- [x] Persistent storage (PostgreSQL, Redis)
- [x] API deployment with HPA
- [x] Worker deployments with scaling
- [x] Service definitions and networking
- [x] Ingress with TLS termination
- [x] Network policies for security

#### ✅ 5. Monitoring & Observability
- [x] Prometheus ServiceMonitors
- [x] PrometheusRules for alerting
- [x] Grafana dashboards
- [x] Application metrics collection
- [x] Log aggregation setup

#### ✅ 6. Security
- [x] Network policies for micro-segmentation
- [x] Security contexts (non-root, read-only filesystem)
- [x] Pod security standards
- [x] Secret management
- [x] RBAC configuration

#### ✅ 7. Backup & Disaster Recovery
- [x] Database backup automation
- [x] Redis state backup
- [x] Kubernetes configuration backup
- [x] Application data backup
- [x] Disaster recovery procedures
- [x] Backup retention policies

#### ✅ 8. Automation & Management
- [x] Automated deployment scripts
- [x] Infrastructure management tools
- [x] Health monitoring scripts
- [x] Scaling automation
- [x] Maintenance procedures

## 📋 Deployment Steps

### 1. Prepare Infrastructure

```bash
# Clone repository
git clone <repository-url>
cd fynqAI.com

# Make scripts executable
chmod +x scripts/*.sh

# Set environment variables
export DOCKER_REGISTRY="your-registry.com"
export DOMAIN_NAME="api.fynqai.com"
export DATABASE_PASSWORD="your-secure-password"
export REDIS_PASSWORD="your-redis-password"
export JWT_SECRET="your-jwt-secret"
```

### 2. Build and Push Images

```bash
# Build production images
./scripts/deploy.sh build

# Push to registry
./scripts/deploy.sh push
```

### 3. Deploy to Kubernetes

```bash
# Deploy infrastructure
./scripts/deploy.sh deploy

# Check deployment status
./scripts/manage.sh status
```

### 4. Configure Monitoring

```bash
# Deploy monitoring stack
kubectl apply -f infrastructure/monitoring/prometheus.yaml

# Access Grafana dashboard
./scripts/manage.sh monitor
```

### 5. Setup Backups

```bash
# Configure backup schedule
./scripts/backup.sh backup-full

# Test disaster recovery
./scripts/backup.sh disaster-recovery backup-file.tar.gz
```

## 🔧 Configuration

### Environment Variables

#### Required Variables
```bash
# Database
DATABASE_URL=postgresql://fynqai:password@postgres:5432/fynqai
DATABASE_PASSWORD=your-secure-password

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=your-redis-password

# Security
JWT_SECRET=your-jwt-secret-key
SECRET_KEY=your-app-secret-key

# External Services
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
```

#### Optional Variables
```bash
# Performance
WORKER_CONCURRENCY=4
API_WORKERS=2
CACHE_TTL=3600

# Features
ENABLE_ANALYTICS=true
ENABLE_FEEDBACK=true
RATE_LIMIT_ENABLED=true

# Backup
BACKUP_BUCKET=fynqai-backups
BACKUP_RETENTION_DAYS=30
```

### Kubernetes Resources

#### Resource Requirements
```yaml
# API Pods
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"

# Worker Pods
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "512Mi"
    cpu: "250m"

# Database
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

#### Scaling Configuration
```yaml
# Horizontal Pod Autoscaler
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 📊 Monitoring & Alerting

### Key Metrics

#### Application Metrics
- API response time (p95, p99)
- Request rate (RPS)
- Error rate (4xx, 5xx)
- Active connections
- Queue depth (Celery)
- Worker task completion rate

#### Infrastructure Metrics
- CPU utilization
- Memory usage
- Disk I/O
- Network throughput
- Pod restart count
- Node resource utilization

### Alert Conditions

#### Critical Alerts
- API error rate > 5%
- Database connection failures
- Pod crash loop backoff
- Disk space > 85%
- Memory usage > 90%

#### Warning Alerts
- API response time > 2s (p95)
- Queue depth > 1000 tasks
- CPU usage > 80%
- Memory usage > 80%

### Grafana Dashboards

1. **Application Overview**
   - Request rate and response times
   - Error rates and status codes
   - Queue metrics and worker status

2. **Infrastructure Health**
   - Node and pod resource usage
   - Network and storage metrics
   - Kubernetes cluster status

3. **Business Metrics**
   - User activity and engagement
   - Feature usage statistics
   - Performance trends

## 🔒 Security Considerations

### Network Security
- TLS encryption for all external traffic
- Internal service mesh encryption
- Network policies for micro-segmentation
- Rate limiting and DDoS protection

### Application Security
- Input validation and sanitization
- SQL injection prevention
- XSS protection headers
- CSRF token validation
- Secure session management

### Infrastructure Security
- Pod security contexts (non-root)
- Read-only root filesystems
- Resource limits and quotas
- Secrets management
- Regular security updates

### Access Control
- RBAC for Kubernetes access
- Service account restrictions
- API authentication and authorization
- Audit logging

## 🚨 Troubleshooting

### Common Issues

#### Pod Startup Issues
```bash
# Check pod status
kubectl get pods -n fynqai

# Describe problematic pod
kubectl describe pod <pod-name> -n fynqai

# Check logs
kubectl logs <pod-name> -n fynqai --previous
```

#### Database Connection Issues
```bash
# Check database pod
kubectl get pods -n fynqai -l app=fynqai-postgres

# Test connection
kubectl exec -it deployment/fynqai-postgres -n fynqai -- psql -U fynqai -d fynqai

# Check database logs
kubectl logs deployment/fynqai-postgres -n fynqai
```

#### Performance Issues
```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n fynqai

# Check HPA status
kubectl get hpa -n fynqai

# Check application metrics
./scripts/manage.sh performance
```

### Recovery Procedures

#### Rolling Back Deployment
```bash
# Check deployment history
kubectl rollout history deployment/fynqai-api -n fynqai

# Rollback to previous version
kubectl rollout undo deployment/fynqai-api -n fynqai

# Check rollback status
kubectl rollout status deployment/fynqai-api -n fynqai
```

#### Database Recovery
```bash
# Create database backup
./scripts/backup.sh backup-db

# Restore from backup
./scripts/backup.sh restore-db backup-file.sql.gz

# Full disaster recovery
./scripts/backup.sh disaster-recovery full-backup.tar.gz
```

## 📈 Performance Optimization

### Application Optimization
- Database query optimization
- Redis caching strategies
- Connection pooling
- Async task processing
- Response compression

### Infrastructure Optimization
- Resource request tuning
- HPA threshold adjustment
- Node affinity rules
- Pod disruption budgets
- Storage optimization

### Cost Optimization
- Right-sizing resources
- Spot instance usage
- Storage class optimization
- Backup retention policies
- Monitoring cost analysis

## 🔄 Maintenance

### Regular Tasks
- [ ] Weekly backup verification
- [ ] Monthly security updates
- [ ] Quarterly capacity planning
- [ ] Annual disaster recovery testing

### Update Procedures
1. Test updates in staging environment
2. Create full backup before updates
3. Rolling update deployment
4. Verify functionality post-update
5. Monitor for issues

### Capacity Planning
- Monitor resource trends
- Plan for traffic growth
- Scale infrastructure proactively
- Optimize resource allocation

## 📞 Support

### Monitoring Access
- Grafana: `kubectl port-forward svc/grafana -n monitoring 3000:3000`
- Prometheus: `kubectl port-forward svc/prometheus -n monitoring 9090:9090`

### Management Commands
```bash
# Infrastructure status
./scripts/manage.sh status

# Application logs
./scripts/manage.sh logs api 100

# Scale services
./scripts/manage.sh scale api 5

# Health check
./scripts/manage.sh health

# Troubleshooting
./scripts/manage.sh troubleshoot
```

### Emergency Contacts
- DevOps Team: devops@fynqai.com
- On-call Engineer: +1-XXX-XXX-XXXX
- Incident Response: incidents@fynqai.com

---

**Production Deployment Complete ✅**

Your fynqAI application is now production-ready with:
- ✅ High availability and scalability
- ✅ Comprehensive monitoring and alerting
- ✅ Automated backup and disaster recovery
- ✅ Security hardening and compliance
- ✅ Performance optimization
- ✅ Operational excellence
