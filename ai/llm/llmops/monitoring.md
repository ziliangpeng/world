# LLM Monitoring and Observability

Effective monitoring is essential for successful LLM training and deployment. With multi-week training runs costing millions and production systems serving millions of users, comprehensive observability prevents catastrophic failures, enables rapid debugging, and ensures system health. This document covers practical monitoring strategies from training through production.

---

## Why Monitoring Matters

### The Cost of Poor Monitoring

| Failure Scenario | Without Monitoring | With Monitoring | Cost Difference |
|------------------|-------------------|-----------------|-----------------|
| **Training divergence** | Discovered after 1 week | Detected in 30 minutes | $200K saved |
| **GPU failure** | 10% degradation for 3 days | Detected immediately | $50K saved |
| **OOM in production** | 1-hour downtime | Auto-scaling triggered | $100K revenue saved |
| **Model quality regression** | Deployed to 100% users | Caught in A/B test | Trust not lost |

### Observability Principles

**Four pillars of observability**:
1. **Metrics**: Quantitative measurements over time (loss, throughput, latency)
2. **Logs**: Discrete events with context (errors, warnings, state changes)
3. **Traces**: Request flow through distributed systems (inference pipeline)
4. **Profiles**: Resource usage patterns (GPU utilization, memory)

---

## Training Monitoring

### 1. Core Training Metrics

**Must-track metrics**:

```python
training_metrics = {
    # Loss & Convergence
    'train_loss': loss.item(),
    'val_loss': val_loss.item(),
    'perplexity': torch.exp(val_loss).item(),
    'gradient_norm': total_norm,

    # Training Dynamics
    'learning_rate': scheduler.get_last_lr()[0],
    'gradient_scale': scaler.get_scale(),  # For mixed precision

    # Progress
    'step': current_step,
    'epoch': current_epoch,
    'tokens_processed': tokens_seen,

    # Throughput
    'samples_per_second': batch_size / step_time,
    'tokens_per_second': batch_size * seq_len / step_time,
    'mfu': model_flops_utilization,  # Model FLOPS Utilization

    # Time
    'step_time': step_time,
    'data_loading_time': data_time,
    'forward_time': forward_time,
    'backward_time': backward_time,
    'optimizer_time': optim_time,
}
```

**Logging every step**:
```python
import wandb

wandb.init(project='llm-training', name='llama-70b-run-1')

for step in range(max_steps):
    metrics = train_step()

    # Log every step
    wandb.log(metrics, step=step)

    # Log detailed metrics less frequently
    if step % 100 == 0:
        detailed_metrics = {
            **metrics,
            'gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,
            'gpu_memory_reserved': torch.cuda.memory_reserved() / 1e9,
            'grad_norm_by_layer': compute_layer_grad_norms(model),
        }
        wandb.log(detailed_metrics, step=step)
```

### 2. Loss Curve Monitoring

**What to watch**:

```python
def check_training_health(loss_history, step):
    """Automated health checks on loss curve"""

    # 1. Loss is decreasing (smoothed)
    recent_loss = np.mean(loss_history[-100:])
    older_loss = np.mean(loss_history[-500:-100])
    if recent_loss > older_loss * 1.1:
        alert("Loss not decreasing", severity='warning')

    # 2. No sudden spikes
    if loss_history[-1] > np.mean(loss_history[-100:]) * 2:
        alert("Loss spike detected", severity='critical')

    # 3. Not diverging (NaN/Inf)
    if np.isnan(loss_history[-1]) or np.isinf(loss_history[-1]):
        alert("Training diverged (NaN/Inf)", severity='critical')
        checkpoint_and_stop()

    # 4. Validation not diverging from training
    val_loss = loss_history['val'][-1]
    train_loss = loss_history['train'][-1]
    if val_loss > train_loss * 1.5:
        alert("Overfitting detected", severity='warning')
```

**Visualization best practices**:
```python
# Log-scale for loss (better visibility)
wandb.log({'train_loss': loss, 'train_loss_log': np.log10(loss)})

# Smoothed curves (exponential moving average)
ema_loss = 0.9 * ema_loss + 0.1 * loss
wandb.log({'loss_ema': ema_loss})

# Multiple validation sets (per-domain tracking)
for domain in ['web', 'code', 'math', 'books']:
    val_loss = evaluate(model, val_sets[domain])
    wandb.log({f'val_loss_{domain}': val_loss})
```

### 3. Hardware Monitoring

**GPU utilization**:
```python
import pynvml

pynvml.nvmlInit()

def log_gpu_metrics():
    """Log GPU health every step"""
    metrics = {}

    for gpu_id in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics[f'gpu_{gpu_id}_util'] = util.gpu
        metrics[f'gpu_{gpu_id}_mem_util'] = util.memory

        # Memory
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        metrics[f'gpu_{gpu_id}_mem_used_gb'] = mem.used / 1e9
        metrics[f'gpu_{gpu_id}_mem_free_gb'] = mem.free / 1e9

        # Temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        metrics[f'gpu_{gpu_id}_temp'] = temp

        # Power
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
        metrics[f'gpu_{gpu_id}_power_w'] = power

    return metrics
```

**Alerts on hardware issues**:
```python
def check_gpu_health(metrics):
    """Alert on GPU problems"""

    for gpu_id in range(num_gpus):
        # Low utilization (possible stall)
        if metrics[f'gpu_{gpu_id}_util'] < 50:
            alert(f"GPU {gpu_id} utilization <50%", severity='warning')

        # High temperature
        if metrics[f'gpu_{gpu_id}_temp'] > 85:
            alert(f"GPU {gpu_id} temperature >85°C", severity='critical')

        # Memory leak detection
        if metrics[f'gpu_{gpu_id}_mem_used_gb'] > expected_memory * 1.2:
            alert(f"GPU {gpu_id} memory leak suspected", severity='warning')
```

### 4. Distributed Training Monitoring

**Synchronization overhead**:
```python
import torch.distributed as dist

def monitor_distributed_training():
    """Track distributed training health"""

    # All-reduce timing (communication overhead)
    start = time.time()
    dist.all_reduce(dummy_tensor)
    comm_time = time.time() - start

    # Straggler detection
    rank_times = [torch.tensor(step_time).cuda() for _ in range(world_size)]
    dist.all_gather(rank_times, torch.tensor(step_time).cuda())

    slowest_rank = torch.argmax(rank_times)
    fastest_rank = torch.argmin(rank_times)
    slowdown = rank_times[slowest_rank] / rank_times[fastest_rank]

    if slowdown > 1.2:
        alert(f"Straggler detected: rank {slowest_rank} is {slowdown:.2f}x slower")

    return {
        'comm_time': comm_time,
        'slowest_rank': slowest_rank.item(),
        'rank_time_variance': torch.std(rank_times).item(),
    }
```

**NCCL debugging**:
```bash
# Enable NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Monitor NCCL operations
export NCCL_LAUNCH_MODE=PARALLEL
```

### 5. Data Pipeline Monitoring

**Data loading bottlenecks**:
```python
def monitor_data_pipeline():
    """Track data loading performance"""

    # Time breakdown
    start_total = time.time()

    start_load = time.time()
    batch = next(dataloader)
    load_time = time.time() - start_load

    start_to_gpu = time.time()
    batch = batch.to('cuda')
    to_gpu_time = time.time() - start_to_gpu

    total_time = time.time() - start_total

    # Alert if data loading is bottleneck
    if load_time > forward_time + backward_time:
        alert("Data loading slower than training", severity='warning')

    return {
        'data_load_time': load_time,
        'data_to_gpu_time': to_gpu_time,
        'data_load_pct': load_time / total_time * 100,
    }
```

**Dataset position tracking**:
```python
# Track where we are in dataset (for resumption)
wandb.log({
    'dataset_epoch': dataloader.epoch,
    'dataset_position': dataloader.get_position(),
    'dataset_samples_seen': total_samples,
})
```

---

## Production Monitoring

### 1. Inference Metrics

**Request-level metrics**:
```python
@app.route('/generate', methods=['POST'])
def generate():
    start_time = time.time()

    # Input
    input_text = request.json['prompt']
    input_tokens = len(tokenizer.encode(input_text))

    # Generation
    with torch.inference_mode():
        output = model.generate(input_text, max_new_tokens=512)

    # Output
    output_tokens = len(tokenizer.encode(output))
    latency = time.time() - start_time

    # Log metrics
    metrics.log({
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
        'latency_ms': latency * 1000,
        'throughput_tokens_per_sec': output_tokens / latency,
        'time_to_first_token_ms': ttft * 1000,  # Measured separately
    })

    return {'response': output, 'latency_ms': latency * 1000}
```

**Aggregated metrics**:
```python
# Track distributions, not just averages
metrics_to_track = {
    'latency_p50': np.percentile(latencies, 50),
    'latency_p95': np.percentile(latencies, 95),
    'latency_p99': np.percentile(latencies, 99),
    'latency_max': np.max(latencies),

    'throughput_p50': np.percentile(throughputs, 50),
    'throughput_p95': np.percentile(throughputs, 95),

    'requests_per_second': len(requests) / time_window,
    'tokens_per_second': sum(total_tokens) / time_window,
}
```

### 2. Model Quality Monitoring

**Online evaluation**:
```python
def continuous_evaluation(model, production_traffic):
    """Evaluate model on real traffic"""

    # Sample requests for evaluation (1-10%)
    eval_sample = random.sample(production_traffic, k=int(len(production_traffic) * 0.05))

    # Run on fixed test set
    test_results = evaluate_on_benchmark(model, test_set='mmlu_sample')

    # User feedback
    thumbs_up_rate = sum(feedback['thumbs_up']) / len(feedback)
    regeneration_rate = sum(feedback['regenerated']) / len(feedback)

    wandb.log({
        'prod_mmlu': test_results['accuracy'],
        'thumbs_up_rate': thumbs_up_rate,
        'regeneration_rate': regeneration_rate,
    })

    # Alert on regression
    if test_results['accuracy'] < baseline_accuracy - threshold:
        alert("Model quality regression detected", severity='critical')
```

**Automated regression detection**:
```python
# Daily evaluation on fixed test set
@scheduler.scheduled_job('interval', hours=24)
def daily_quality_check():
    """Run daily quality check"""
    scores = evaluate_benchmark_suite(model)

    for benchmark, score in scores.items():
        # Compare to baseline
        baseline = baseline_scores[benchmark]
        drop = baseline - score

        if drop > regression_threshold:
            alert(
                f"Regression on {benchmark}: {score:.2f} vs baseline {baseline:.2f}",
                severity='critical'
            )

            # Automatically rollback if severe
            if drop > critical_threshold:
                rollback_to_previous_model()
```

### 3. System Health Monitoring

**Resource utilization**:
```python
import psutil

def monitor_system_resources():
    """Monitor CPU, memory, disk, network"""

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()

    # Memory
    mem = psutil.virtual_memory()
    mem_used_gb = mem.used / 1e9
    mem_percent = mem.percent

    # Disk
    disk = psutil.disk_usage('/')
    disk_used_gb = disk.used / 1e9
    disk_percent = disk.percent

    # Network
    net = psutil.net_io_counters()
    net_sent_mb = net.bytes_sent / 1e6
    net_recv_mb = net.bytes_recv / 1e6

    return {
        'cpu_percent': cpu_percent,
        'memory_used_gb': mem_used_gb,
        'memory_percent': mem_percent,
        'disk_used_gb': disk_used_gb,
        'disk_percent': disk_percent,
        'network_sent_mb': net_sent_mb,
        'network_recv_mb': net_recv_mb,
    }
```

**Alerts on resource issues**:
```python
def check_system_health(metrics):
    """Alert on system resource problems"""

    if metrics['memory_percent'] > 90:
        alert("Memory usage >90%", severity='critical')

    if metrics['disk_percent'] > 85:
        alert("Disk usage >85%", severity='warning')

    if metrics['cpu_percent'] > 95:
        alert("CPU usage >95%", severity='warning')
```

### 4. Error Rate Monitoring

**Track errors by type**:
```python
from collections import Counter
import traceback

error_counts = Counter()

@app.errorhandler(Exception)
def handle_error(error):
    """Log all errors"""

    error_type = type(error).__name__
    error_counts[error_type] += 1

    # Log to monitoring
    metrics.increment('errors_total')
    metrics.increment(f'errors_{error_type}')

    # Log full traceback
    logger.error(f"Error: {error}", exc_info=True)

    # Alert on high error rate
    if error_counts.total() / requests.total() > 0.01:  # >1% error rate
        alert("Error rate >1%", severity='critical')

    return {'error': str(error)}, 500
```

**Error rate SLOs**:
```python
# Service Level Objectives
SLO_TARGETS = {
    'availability': 0.999,  # 99.9% uptime
    'error_rate': 0.001,  # <0.1% errors
    'latency_p95': 2000,  # p95 <2s
    'latency_p99': 5000,  # p99 <5s
}

def check_slo_compliance(metrics):
    """Check if SLOs are met"""

    # Availability
    uptime = metrics['successful_requests'] / metrics['total_requests']
    if uptime < SLO_TARGETS['availability']:
        alert(f"SLO violation: availability {uptime:.4f}", severity='critical')

    # Error rate
    error_rate = metrics['errors'] / metrics['total_requests']
    if error_rate > SLO_TARGETS['error_rate']:
        alert(f"SLO violation: error rate {error_rate:.4f}", severity='critical')

    # Latency
    if metrics['latency_p95'] > SLO_TARGETS['latency_p95']:
        alert(f"SLO violation: p95 latency {metrics['latency_p95']}ms", severity='warning')
```

---

## Alerting Strategies

### 1. Alert Severity Levels

```python
class AlertSeverity:
    INFO = 1      # Informational, no action needed
    WARNING = 2   # Should investigate, not urgent
    ERROR = 3     # Action needed soon
    CRITICAL = 4  # Immediate action required

alert_configs = {
    # Training alerts
    'loss_spike': {'severity': CRITICAL, 'cooldown': 300},  # 5min cooldown
    'loss_not_decreasing': {'severity': WARNING, 'cooldown': 3600},
    'gpu_low_util': {'severity': WARNING, 'cooldown': 1800},
    'training_diverged': {'severity': CRITICAL, 'cooldown': 0},

    # Production alerts
    'high_latency': {'severity': WARNING, 'cooldown': 600},
    'error_rate_high': {'severity': CRITICAL, 'cooldown': 300},
    'model_regression': {'severity': CRITICAL, 'cooldown': 0},
    'out_of_memory': {'severity': CRITICAL, 'cooldown': 0},
}
```

### 2. Smart Alerting (Avoid Fatigue)

```python
class AlertManager:
    def __init__(self):
        self.last_alert_time = {}
        self.alert_counts = Counter()

    def alert(self, message, severity, cooldown=300):
        """Send alert with cooldown to avoid spam"""

        # Check cooldown
        if message in self.last_alert_time:
            time_since = time.time() - self.last_alert_time[message]
            if time_since < cooldown:
                return  # Skip alert (in cooldown)

        # Send alert
        self._send_alert(message, severity)
        self.last_alert_time[message] = time.time()
        self.alert_counts[message] += 1

        # Meta-alert if too many alerts
        if len(self.alert_counts) > 10:
            self._send_alert(
                f"Alert storm: {len(self.alert_counts)} distinct alerts",
                severity='CRITICAL'
            )

    def _send_alert(self, message, severity):
        """Send to appropriate channel based on severity"""
        if severity == 'CRITICAL':
            pagerduty.trigger(message)  # Wake up on-call
            slack.send('#incidents', message)
        elif severity == 'WARNING':
            slack.send('#monitoring', message)
        else:
            log.info(message)
```

### 3. Alert Channels

```python
# Multi-channel alerting
def send_alert(message, severity):
    """Route alerts to appropriate channels"""

    # Always log
    logger.log(severity, message)

    # Slack
    if severity in ['WARNING', 'ERROR', 'CRITICAL']:
        slack_webhook(
            channel='#llm-monitoring',
            message=message,
            severity=severity
        )

    # PagerDuty (critical only, on-call rotation)
    if severity == 'CRITICAL':
        pagerduty.trigger_incident(
            title=message,
            severity='critical',
            service='llm-training'
        )

    # Email (weekly digest for INFO/WARNING)
    if severity in ['INFO', 'WARNING']:
        email_digest.add(message)  # Sent weekly

    # Datadog event
    datadog.event(
        title=message,
        text=message,
        alert_type=severity.lower(),
        tags=['service:llm', 'env:production']
    )
```

---

## Monitoring Tools & Infrastructure

### 1. Weights & Biases (Training)

```python
import wandb

# Initialize
wandb.init(
    project='llm-training',
    name=f'llama-70b-{run_id}',
    config={
        'model': 'llama-70b',
        'batch_size': 1024,
        'learning_rate': 3e-4,
        'num_gpus': 512,
    }
)

# Log metrics
wandb.log({
    'train_loss': loss,
    'val_loss': val_loss,
    'learning_rate': lr,
    'gpu_memory_gb': mem_gb,
}, step=step)

# Log artifacts
wandb.log_artifact('checkpoint-1000/', type='model')

# Log system metrics (automatic)
wandb.watch(model, log='all', log_freq=100)
```

**W&B features**:
- Real-time metric visualization
- Experiment comparison
- Artifact versioning
- Collaborative dashboards
- Alert integration

### 2. Prometheus + Grafana (Production)

**Prometheus metrics collection**:
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
request_count = Counter('llm_requests_total', 'Total requests')
request_latency = Histogram('llm_request_latency_seconds', 'Request latency')
gpu_memory = Gauge('llm_gpu_memory_used_bytes', 'GPU memory used', ['gpu_id'])
error_count = Counter('llm_errors_total', 'Total errors', ['error_type'])

# Instrument code
@request_latency.time()
def generate(prompt):
    request_count.inc()

    try:
        output = model.generate(prompt)
        return output
    except Exception as e:
        error_count.labels(error_type=type(e).__name__).inc()
        raise

# Update GPU metrics
for gpu_id in range(num_gpus):
    mem_used = torch.cuda.memory_allocated(gpu_id)
    gpu_memory.labels(gpu_id=gpu_id).set(mem_used)

# Start metrics server
start_http_server(8000)  # Prometheus scrapes this
```

**Grafana dashboard**:
```yaml
# Example Grafana dashboard config
panels:
  - title: "Request Rate"
    metric: "rate(llm_requests_total[5m])"
    type: "graph"

  - title: "Latency Percentiles"
    metrics:
      - "histogram_quantile(0.5, llm_request_latency_seconds)"
      - "histogram_quantile(0.95, llm_request_latency_seconds)"
      - "histogram_quantile(0.99, llm_request_latency_seconds)"

  - title: "Error Rate"
    metric: "rate(llm_errors_total[5m]) / rate(llm_requests_total[5m])"
    alert: "> 0.01"  # Alert if >1% errors
```

### 3. Datadog (Full-Stack)

```python
from datadog import initialize, statsd

# Initialize
initialize(api_key='your_key', app_key='your_app_key')

# Metrics
statsd.increment('llm.requests')
statsd.histogram('llm.latency', latency_ms, tags=['model:llama-70b'])
statsd.gauge('llm.gpu.memory', mem_gb, tags=['gpu:0'])

# Events
from datadog import api
api.Event.create(
    title='Model deployed',
    text='Llama-70b-v2 deployed to production',
    tags=['deployment', 'llm'],
)

# APM (tracing)
from ddtrace import tracer

@tracer.wrap()
def generate(prompt):
    return model.generate(prompt)
```

### 4. Custom Dashboards

**Streamlit monitoring dashboard**:
```python
import streamlit as st
import pandas as pd

st.title("LLM Training Monitor")

# Fetch metrics from W&B
runs = wandb.Api().runs("llm-training")
latest_run = runs[0]

# Loss curve
st.line_chart(latest_run.history()['train_loss'])

# Metrics table
metrics_df = pd.DataFrame({
    'Metric': ['Train Loss', 'Val Loss', 'Perplexity', 'GPU Util'],
    'Value': [
        latest_run.summary['train_loss'],
        latest_run.summary['val_loss'],
        latest_run.summary['perplexity'],
        latest_run.summary['gpu_util'],
    ]
})
st.table(metrics_df)

# GPU utilization
st.bar_chart(latest_run.history()['gpu_util'])

# Alerts
if latest_run.summary['train_loss'] > 3.0:
    st.error("⚠️ High training loss detected!")
```

---

## Best Practices

### ✅ Do This

1. **Log aggressively during training**
   - Every step: loss, LR, throughput
   - Every 100 steps: GPU metrics, gradient norms
   - Every eval: full benchmark suite

2. **Use multiple monitoring tools**
   - W&B for training experiments
   - Prometheus+Grafana for production
   - PagerDuty for critical alerts

3. **Monitor distributions, not just averages**
   - p50, p95, p99 latencies (not just mean)
   - Min, max, variance

4. **Set up automated alerts**
   - Training divergence (NaN, loss spike)
   - Hardware failures (GPU down, OOM)
   - Quality regression (benchmark drop)
   - SLO violations (latency, error rate)

5. **Create dashboards for different audiences**
   - Engineers: detailed metrics, traces
   - Managers: high-level KPIs
   - On-call: actionable alerts

6. **Version your metrics**
   - Track which model version generated which metrics
   - Enable rollback correlation

7. **Test your alerts**
   - Regularly trigger test alerts
   - Ensure on-call rotation works

### ❌ Avoid This

1. **Don't just track training loss**
   - Need validation, per-domain, hardware metrics

2. **Don't ignore hardware metrics**
   - GPU failures can be silent (slow degradation)

3. **Don't over-alert**
   - Use cooldowns and thresholds
   - Avoid alert fatigue

4. **Don't forget user-facing metrics**
   - Latency, error rate, user satisfaction

5. **Don't rely on a single monitoring tool**
   - Redundancy matters (monitoring can fail too)

---

## Case Studies

### Case Study 1: Silent GPU Failure

**Problem**: Training throughput decreased 15% over 2 days, not noticed until week later.

**Root cause**:
- 8 out of 512 GPUs degraded (ECC errors)
- Still functional but 50% slower
- Training continued but inefficient

**Solution**:
- Added per-GPU throughput monitoring
- Alert when any GPU <80% of cluster average
- Detected degradation in 30 minutes

**Cost saved**: $100K (2 weeks of degraded training prevented)

### Case Study 2: Production Quality Regression

**Problem**: Model deployed with 5% lower MMLU score, discovered after 1 week in production.

**Root cause**:
- Checkpoint from divergent training run
- No automated quality checks before deployment

**Solution**:
- Daily automated evaluation on fixed test set
- Alert if any benchmark <95% of baseline
- Caught regression in A/B test before full rollout

**Lesson**: Always monitor quality continuously, even in production.

### Case Study 3: Memory Leak in Inference

**Problem**: Inference server OOM after 12 hours of serving.

**Root cause**:
- Slow memory leak in KV cache management
- Memory usage increased 100MB/hour

**Solution**:
- Added memory monitoring dashboard
- Alert when memory grows >10% per hour
- Detected leak in 2 hours vs 12

**Lesson**: Monitor resource trends, not just absolute values.

---

## Resources

### Tools

**Training Monitoring**:
- [Weights & Biases](https://wandb.ai/) - Experiment tracking and visualization
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Metric visualization
- [MLflow](https://mlflow.org/) - Model lifecycle tracking
- [Comet](https://www.comet.com/) - Experiment management

**Production Monitoring**:
- [Prometheus](https://prometheus.io/) - Metrics collection
- [Grafana](https://grafana.com/) - Dashboards and visualization
- [Datadog](https://www.datadoghq.com/) - Full-stack observability
- [New Relic](https://newrelic.com/) - APM and monitoring

**Alerting**:
- [PagerDuty](https://www.pagerduty.com/) - Incident management
- [Opsgenie](https://www.atlassian.com/software/opsgenie) - Alert routing
- [Slack](https://slack.com/) - Team notifications

### Reading

- [Monitoring Machine Learning Models in Production](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/) - Comprehensive guide
- [LLM Observability Best Practices](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/) - Eugene Yan et al.
- [Production ML Monitoring](https://huyenchip.com/2023/04/11/llm-engineering.html#monitoring) - Chip Huyen

---

**Related Documentation**:
- [Evaluation](evaluation.md) - What metrics to monitor for quality
- [Checkpoint Management](checkpoint-management.md) - Monitoring checkpoint health
- [Training Stability](../pre-training/training-stability.md) - Detecting and fixing training issues
- [Distributed Training](../pre-training/distributed-training.md) - Monitoring distributed systems
