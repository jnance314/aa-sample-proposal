# 📝 Product Requirements Document (PRD)

## Phase 1: Project Scaffolding & Simulation Layer



### 🎯 Objective

To develop the **foundation of the system architecture and data generation pipeline** by implementing a realistic, structured, and extensible simulation layer for SMR component telemetry, along with initial project scaffolding. This phase will establish:



1. The base repo and modular code structure for the entire stack.
2. A realistic, parameterized simulation engine that generates **multi-reactor, multi-site telemetry data** for SMR components.
3. Output in both **batch mode** (for offline ML training) and **streaming mode** (for Kafka/real-time consumption).
4. Configuration-driven reactor/component simulation presets.
5. Fully deterministic and reproducible data generation flows for both manual testing and CI validation.

The simulator will act as a **drop-in placeholder for the real digital twin** expected in production, enabling downstream system development (modeling, dashboards, compliance logging) to proceed independently of real hardware.


### 🔍 Background

Applied Atomics and similar early-stage SMR startups are in the **pre-deployment, pre-data** stage. Yet to begin building a production-grade predictive maintenance system, the data and ML engineering pipeline must be developed in parallel with the reactor design.

Since no real sensors exist yet, this simulation engine will:

* Mimic the behavior of heat exchangers, coolant pumps, and other subsystems
* Model realistic degradation and failure progression
* Allow fault injection for training and testing anomaly detection models
* Be parameterized per site/reactor/component

This allows **future integration of digital twin outputs** with minimal friction and ensures that **realistic ML pipelines, dashboards, and audit systems can be prototyped today**.


### 🧱 Scope of Work

#### 1. **Project Directory & Module Scaffold**

**Goal:** Establish the file/folder structure for `pm-stack-smr` with modular subfolders, empty init files, and dev setup instructions.

**Deliverables:**

* `README.md` with project goals, structure, and setup instructions
* `.gitignore`, `requirements.txt`, and either `venv` or `poetry` support
* Top-level directories:
  * `simulation/`, `sensor_adapters/`, `streaming/`, `storage/`, `models/`, `ml_pipeline/`, `dashboard/`, `audit/`, `infra/`, `docs/`


**Dependencies:** None**Estimated Time:** 1 hour


#### 2. **ReactorSimulator Engine**

**Goal:** Build a class-based simulation engine to produce synthetic telemetry representing a degrading SMR component, e.g., a coolant pump.

**Key Features:**

* Cyclical simulation (`simulate_cycle()`) returns structured telemetry:
  * `temperature_in`, `temperature_out`, `pressure_drop`, `vibration_rms`, etc.
* Simulates exponential decay or other failure curves (`performance` or `efficiency`)
* Configurable degradation speed, noise levels, and thresholds
* Deterministic via `random_seed` input

**Output format:**

```python
{
  "timestamp": "2025-04-04T12:01:00Z",
  "site_id": "TX01",
  "reactor_id": "R2",
  "component": "coolant_pump",
  "temperature_in": 225.0,
  "temperature_out": 204.3,
  "flow_rate": 97.2,
  "pressure_drop": 6.1,
  "vibration_rms": 0.02,
  "performance": 0.86,
  "true_RUL": 54
}
```

**Deliverables:**

* `simulation/generators.py` with `ReactorSimulator` class
* Simulation logic for at least one component (coolant pump)
* `__init__.py` and unit test stubs


**Dependencies:** `numpy`, `datetime`, `uuid`**Estimated Time:** 6 hours


#### 3. **Fault Injection Framework**

**Goal:** Provide a method to dynamically inject faults into the simulation timeline, altering the degradation curve mid-simulation.

**Supported fault types:**

* Sudden failure (step degradation)
* Accelerated decay (change in lambda)
* Random signal spikes or drops

**Interface:**

```python
def inject_fault(self, fault_type: str) -> None
```

**Impact:**

* Allows testing of anomaly detection systems
* Simulates real-world faults like coolant leaks, seal wear, etc.

**Deliverables:**

* Extension of `ReactorSimulator` with fault injection
* Associated unit tests with fixed random seeds for reproducibility


**Dependencies:** `random`, `math`, base simulator**Estimated Time:** 3 hours


#### 4. **Simulation CLI Tool**

**Goal:** Build a command-line interface to run the simulator in batch mode (CSV output) or streaming mode (Kafka output, to be implemented later).

**CLI Options:**

* `--site-id`
* `--reactor-id`
* `--cycles` (default: 200)
* `--mode` (`batch` or `stream`)
* `--output-path` for CSV
* `--seed`

**Deliverables:**

* CLI tool in `simulation/run_simulation.py`
* Writes CSV with telemetry for `cycles` iterations
* Stub function to integrate with Kafka (to be filled in Phase 2)


**Dependencies:** `argparse`, `pandas`, `uuid`**Estimated Time:** 2 hours


#### 5. **Component Configuration System**

**Goal:** Allow components to be defined via external YAML config, so new types (e.g., control rods, heat exchangers) can be added declaratively.

**Example structure (**`reactor_config.yaml`):

```yaml
TX01:
  R1:
    coolant_pump:
      decay_type: exponential
      lambda: 0.02
      noise_std: 0.01
      fail_threshold: 0.3
    heat_exchanger:
      decay_type: linear
      slope: 0.001
```

**Usage:**

* `ReactorSimulator` reads config during initialization
* Component-specific behavior is dynamically generated

**Deliverables:**

* `simulation/reactor_config.yaml`
* Config parser in `generators.py`
* Validation against schema


**Dependencies:** `yaml`, `pydantic` or manual schema validation**Estimated Time:** 2 hours


### 🧪 Acceptance Criteria

✅ Repo includes fully functioning `ReactorSimulator` class

✅ Config-driven simulation outputs realistic multi-site data

✅ Fault injection alters degradation and outputs correctly

✅ CLI supports CSV batch mode with deterministic outputs

✅ Unit test scaffolds exist for core simulation components

✅ Output schema matches downstream system expectations


### 🔄 Dependencies & Future Integration

| Dependency | Consumes Phase 1 Output |
|----|----|
| `sensor_adapters/` | Uses `simulate_cycle()` for polling |
| `streaming/kafka_producer.py` | Will consume streaming mode in Phase 2 |
| `ml_pipeline/train.py` | Will use CSVs for model training |
| `dashboard/` | Will visualize metrics from simulation data |
| `audit/` | Will eventually log predictions made on simulated inputs |


### ⏱️ Time Estimate Summary

| Task | Hours |
|----|----|
| Repo structure | 1 |
| Simulator core | 6 |
| Fault injection | 3 |
| CLI runner | 2 |
| Config loader | 2 |
| **Total** | **14 hours** |


### 📌 Out of Scope for Phase 1

* Kafka integration (defined but not implemented here)
* Real-time dashboarding or API serving
* ML model training
* InfluxDB/MLflow storage
* Compliance auditing

These are all downstream dependencies that will hook into this simulation engine.

# 📝 Product Requirements Document (PRD)

## Phase 2: Sensor Adapters & Streaming



### 🎯 Objective

To develop the interface between simulated sensor telemetry and the real-time data ingestion pipeline using a **mocked industrial protocol interface** and a **Kafka-based streaming infrastructure**. This phase will:



1. Simulate a Modbus-compatible sensor adapter that mimics how telemetry would be polled from real SMR hardware (e.g., via PLCs).
2. Establish a Kafka-based message broker for real-time telemetry publishing.
3. Implement Kafka producers that publish telemetry data per cycle.
4. Implement Kafka consumers that receive telemetry data and route it downstream (e.g., to storage, inference, audit modules).
5. Ensure the system is extensible for future multi-site, multi-topic ingestion.

This phase connects the **simulation output from Phase 1** to the rest of the streaming, ML, and monitoring stack.


### 🔍 Background

In industrial SMR systems, sensor telemetry is typically pulled from field devices via protocols like **Modbus**, **OPC UA**, or **Profinet**. For now, we mock this layer but simulate it accurately, so the downstream infrastructure (streaming, dashboards, compliance) can be developed as if connected to live hardware.

We use **Apache Kafka** as the real-time messaging layer, due to its:

* Strong compatibility with high-throughput IoT systems
* Partitioning and replayability
* Integration with time-series DBs, inference APIs, and audit logs

Kafka will serve as the primary ingestion layer for downstream consumers: storage (InfluxDB), ML (gRPC inference), and compliance (logging).


### 🧱 Scope of Work

#### 2.1 **Mock Sensor Interface (Modbus-Like Adapter)**

**Goal:** Simulate an interface that behaves like a low-level polling API used in real industrial hardware. This provides an abstraction over `ReactorSimulator` and helps future-proof the system for real sensor integration.

**Features:**

* Public methods like `read_temperature()`, `read_flow_rate()`, etc.
* Internally pulls from `ReactorSimulator.simulate_cycle()` once per poll
* Adds random latency, optional jitter, and fault injection

**Interface:**

```python
class MockModbusAdapter:
    def __init__(self, simulator: ReactorSimulator)
    def read_temperature(self) -> float
    def read_pressure(self) -> float
    def read_flow_rate(self) -> float
    def read_vibration(self) -> float
```

**Deliverables:**

* `sensor_adapters/modbus_mock.py`
* Adapter test harness
* Minimal CLI for manual polling


**Dependencies:** `simulation/generators.py`**Estimated Time:** 3 hours


#### 2.2 **Kafka Producer Implementation**

**Goal:** Publish simulated telemetry to a Kafka topic (`telemetry_raw`) on a per-cycle basis, simulating a real-time data stream from hardware sensors.

**Features:**

* Publishes JSON messages every N seconds (default: 1 Hz)
* Kafka producer uses a structured schema with input validation
* Supports publishing from multiple simulators (multi-reactor)
* Includes metadata (site ID, reactor ID, component, timestamp)

**Example message:**

```json
{
  "site_id": "TX01",
  "reactor_id": "R1",
  "component": "heat_exchanger",
  "timestamp": "2025-04-04T12:02:00Z",
  "metrics": {
    "temperature_in": 225.3,
    "temperature_out": 204.7,
    "pressure_drop": 6.5,
    "flow_rate": 93.2
  }
}
```

**Interface:**

```python
def publish_telemetry(simulator: ReactorSimulator, topic: str, interval_sec: float)
```

**Deliverables:**

* `streaming/kafka_producer.py`
* Integration with `MockModbusAdapter`
* Logging and retry logic


**Dependencies:** `confluent_kafka`, `pydantic` (for schema validation)**Estimated Time:** 4 hours


#### 2.3 **Kafka Consumer Implementation**

**Goal:** Consume telemetry messages from Kafka and route them to downstream consumers: storage (InfluxDB), ML inference (gRPC), and audit logging.

**Features:**

* Subscribes to `telemetry_raw`
* Extracts data, transforms to required formats
* Calls:
  * `InfluxWriter.write_point()` (Phase 3)
  * `RULInferenceClient.predict()` (Phase 5)
  * `AuditLogger.log_prediction()` (Phase 7)

**Interface:**

```python
def consume_telemetry(topic: str):
    # while True:
    #   read message
    #   extract features
    #   send to downstream sinks
```

Supports modular routing based on message contents (e.g., component type, site ID).

**Deliverables:**

* `streaming/kafka_consumer.py`
* Message parsing, schema validation, transformation
* Graceful failure handling, logging


**Dependencies:** `confluent_kafka`, `pydantic`, `jsonschema`**Estimated Time:** 5 hours


#### 2.4 **Docker Compose Setup for Kafka/Zookeeper**

**Goal:** Enable local development and testing with a fully containerized Kafka + Zookeeper environment.

**Features:**

* Single-node Kafka broker
* Default topic: `telemetry_raw`
* Script to create topics and verify health
* Healthcheck and retry support for producer and consumer

**Deliverables:**

* `infra/docker-compose.yml` (add Kafka + Zookeeper services)
* `infra/init_kafka_topics.sh`
* Docs for local setup and connectivity test script


**Dependencies:** Docker, `kafka-python` or `confluent-kafka`**Estimated Time:** 2 hours


### 📈 Metrics of Success

* Producer and consumer can stream and receive messages without failure.
* Message structure is valid and matches simulation schema.
* Messages are routed to the correct downstream stub (Influx, inference).
* Adapter mimics a realistic Modbus-style API.
* Local Kafka setup works reliably with logs and retry logic.


### ⏱️ Time Estimate Summary

| Task | Description | Est. Hours |
|----|----|----|
| 2.1 | Sensor adapter (Modbus-like) | 3 hrs |
| 2.2 | Kafka producer | 4 hrs |
| 2.3 | Kafka consumer | 5 hrs |
| 2.4 | Docker + Kafka setup | 2 hrs |
| **Total** | — | **14 hours** |


### 🔄 Dependencies & Downstream Integrations

| Dependency | Description |
|----|----|
| ✅ **Upstream**: `ReactorSimulator` | Adapter wraps the simulator |
| ✅ **Downstream**: `storage/influx_logger.py` | Consumer calls this in Phase 3 |
| ✅ **Downstream**: `ml_pipeline/grpc_client.py` | Used in Phase 5 |
| ✅ **Downstream**: `audit/prediction_logger.py` | Used in Phase 7 |

This phase **unlocks all downstream modules**, as it formalizes the ingestion point and routing logic of the system.


### 📌 Out of Scope

* Real Modbus or OPC UA protocol integration (planned for future real hardware)
* Time-series database storage (InfluxDB handled in Phase 3)
* Real-time ML inference (handled in Phase 5)
* Front-end visualization (handled in Phase 6)


### 🧪 Test Plan

| Component | Test Case | Expected Result |
|----|----|----|
| Adapter | `.read_temperature()` | Returns float within expected range |
| Producer | Publishes simulated cycle | Message is written to Kafka topic |
| Consumer | Reads Kafka message | Message parsed and dispatched to mock sink |
| Docker | `docker-compose up` | Kafka/Zookeeper available, topics initialized |

# 📝 Product Requirements Document (PRD)

## Phase 3: Storage Systems



### 🎯 Objective

To implement the **data persistence infrastructure** that underpins both real-time and offline operations for the predictive maintenance system. This includes:



1. A **time-series database (InfluxDB)** for telemetry data visualization and streaming analysis (via Grafana).
2. A **batch data writer** that stores structured, feature-engineered data for ML training in **Parquet format**.
3. Integration of the **MLflow model registry**, which will track all models, metrics, and parameters used during model development.
4. Utilities to **log and retrieve models** from MLflow in a reproducible, scalable, and compliant manner.

This phase is essential for downstream analytics, model training, audit logging, and MLOps readiness.


### 🔍 Background

The system is designed to eventually operate with real SMR telemetry from multiple sites. Until then, **simulated telemetry** (via `ReactorSimulator` and the Kafka pipeline) needs to be captured and persisted reliably in multiple formats, for different consumers:

* **Time-series format** (InfluxDB) to support operator dashboards and real-time visualization.
* **Batch tabular format** (Parquet) for ML training pipelines, including full feature sets.
* **ML model metadata and versioning** to be tracked and persisted via **MLflow**, enabling:
  * Reproducibility
  * Compliance
  * Comparison between experiments
  * Controlled model rollout to inference APIs

Each of these subsystems must be cleanly decoupled, easily tested, and extensible to new sensor types or ML use cases.


### 🧱 Scope of Work

#### 3.1 **Time-Series Storage (InfluxDB Integration)**

**Goal:** Store telemetry data in a time-series database for later visualization in Grafana and query via dashboards.

**Features:**

* Write points to InfluxDB from streaming Kafka consumer
* Tag data with `site_id`, `reactor_id`, and `component`
* Organize fields as sensor measurements (e.g. `temperature_in`, `pressure_drop`)
* Support high-frequency ingestion (e.g., 1 Hz)
* Handle connection loss, retries, and timestamp collisions

**Interface:**

```python
class InfluxWriter:
    def __init__(self, db_url: str, token: str, org: str, bucket: str)
    def write_point(self, sensor_data: dict)
    def flush(self)
```

**Data Example:**

```plaintext
measurement="telemetry"
tags: site_id=TX01, reactor_id=R2, component=coolant_pump
fields: temperature_in=225.0, pressure_drop=5.9, flow_rate=96.5
timestamp: 2025-04-04T12:00:00Z
```

**Deliverables:**

* `storage/influx_logger.py`
* Initialization scripts and test ingestion
* Integration in Kafka consumer


**Dependencies:** `influxdb-client`, InfluxDB 2.x, Docker container**Estimated Time:** 3 hours


#### 3.2 **Batch Data Storage for ML Training (Parquet Writer)**

**Goal:** Persist structured feature-labeled datasets for use in ML training and validation workflows. Supports consistent input to the RUL model.

**Features:**

* Accepts sensor snapshots and engineered features
* Appends to partitioned Parquet file
* Stores `true_RUL` and `component_status` (for training targets)
* Supports offline simulation ingestion or batch conversion from streaming logs

**Interface:**

```python
def write_training_data(data: pd.DataFrame, path: str, partition_cols=["site_id", "component"])
```

**Schema:**

| site_id | reactor_id | component | cycle | temperature_in | pressure_drop | RUL |
|----|----|----|----|----|----|----|
| TX01 | R1 | pump | 42 | 225.1 | 6.4 | 58 |

**Deliverables:**

* `storage/parquet_writer.py`
* CLI or callable script to convert CSV/Kafka output into batch files
* Example dataset (`training_sample.parquet`)


**Dependencies:** `pandas`, `pyarrow`, `fastparquet`**Estimated Time:** 2 hours


#### 3.3 **MLflow Tracking Server Setup**

**Goal:** Stand up a locally hosted (or Dockerized) **MLflow Tracking Server** to persist models, hyperparameters, and training metadata. Will be used throughout the pipeline to track and version all ML artifacts.

**Features:**

* Support local or cloud artifact store (default: local filesystem)
* Secure endpoint (if hosted on GCP later)
* Organized experiment tracking by model type
* Integration with `train.py`, `validate.py`, and inference API

**Setup:**

* Docker Compose service for local deployment
* Environment variables for GCP-compatible storage later (e.g. GCS bucket)
* Local SQLite or Postgres backend store

**Deliverables:**

* MLflow Tracking Server running on `localhost:5000`
* Preconfigured with default experiment
* Integration doc for downstream developers


**Dependencies:** `mlflow`, Docker, optional GCP SDK**Estimated Time:** 2 hours


#### 3.4 **MLflow Utilities for Model Logging and Loading**

**Goal:** Provide wrappers and helper functions that standardize how models are saved, logged, registered, and loaded across the codebase.

**Functions:**

```python
def log_model(model, model_name, params, metrics, artifacts):
    # Log to MLflow and register version

def load_model(model_uri: str):
    # Load model from local or remote MLflow server
```

**Logging Includes:**

* Training parameters
* Input schema
* Evaluation metrics (MAE, RMSE, etc.)
* SHAP feature importance plots (if available)
* Feature engineering config hash

**Deliverables:**

* `storage/mlflow_registry.py`
* Example in `train.py` and `validate.py`
* Integration test to ensure round-trip save/load


**Dependencies:** `mlflow`, `matplotlib`, `json`**Estimated Time:** 3 hours


### 📈 Metrics of Success

| Target | Measurable Outcome |
|----|----|
| Time-series storage | 100+ telemetry points stored in InfluxDB and visible in Grafana |
| Batch dataset | Parquet file created from simulation with RUL column present |
| MLflow | Server starts via Docker and logs first model |
| Logging | ML model, metrics, and config saved and reloadable |


### ⏱️ Time Estimate Summary

| Task | Description | Est. Hours |
|----|----|----|
| 3.1 | Time-series DB + InfluxWriter | 3 hrs |
| 3.2 | Parquet training data writer | 2 hrs |
| 3.3 | MLflow Tracking Server setup | 2 hrs |
| 3.4 | MLflow utilities for logging/loading | 3 hrs |
| **Total** | — | **10 hours** |


### 🔄 Dependencies & Downstream Integrations

| Dependency | Description |
|----|----|
| ✅ **Upstream**: `kafka_consumer.py` | Passes data to `InfluxWriter` |
| ✅ **Upstream**: `feature_engineering.py` | Outputs features for batch writer |
| ✅ **Downstream**: `train.py`, `validate.py` | Uses Parquet files + logs to MLflow |
| ✅ **Downstream**: `grpc_server.py` | Loads models from MLflow for inference |
| ✅ **Downstream**: `audit_logger.py` | Logs model version info for traceability |


### 📌 Out of Scope

* Feature engineering logic (already defined in `feature_engineering.py`)
* Training scripts or model development
* Grafana dashboards (handled in Phase 6)
* Audit hooks (Phase 7)
* CI/CD integration for storage or model deployment (future enhancement)


### 🧪 Test Plan

| Component | Test Case | Expected Output |
|----|----|----|
| `InfluxWriter` | Writes a telemetry point | Record appears in InfluxDB with correct tags |
| `ParquetWriter` | Writes 500 rows of training data | File loads successfully with full schema |
| MLflow Server | Starts via Docker | UI reachable at `localhost:5000` |
| MLflow Log/Load | Save and reload model | Returns identical predictions |

# 📝 Product Requirements Document (PRD)

## Phase 4: ML Modeling and Feature Engineering



### 🎯 Objective

To develop the predictive models and feature pipelines necessary for accurate **Remaining Useful Life (RUL)** estimation and **anomaly detection** of SMR components, based on simulated or real sensor telemetry.

This includes:



1. A **feature engineering pipeline** that transforms raw time-series sensor data into model-ready sequences and aggregates.
2. A **deep learning-based RUL model** (LSTM or Bi-LSTM) using PyTorch.
3. A **statistical/ML anomaly detection module** using methods like Isolation Forest or PCA.
4. Training and evaluation workflows with full logging to **MLflow**, producing versioned, explainable, and traceable model artifacts.

This phase establishes your **ML system foundation**, and proves your ability to:

* Turn domain knowledge into feature representations
* Select and implement suitable model architectures
* Establish a reproducible experimentation loop
* Ship reliable models ready for production inference and compliance


### 🔍 Background

RUL estimation and fault detection are **core predictive maintenance tasks**. While the techniques themselves are well-known, the *execution*—in a way that’s reproducible, explainable, and architecture-compatible—is the differentiator.

These models will power downstream components like:

* gRPC inference server
* Ops dashboards (RUL curves, fault flags)
* Compliance and audit trails (model versioning, input tracing)

The modeling stack must:

* Accept structured input from the simulation or real systems
* Be robust to noise and partial observability
* Be able to generalize across multiple reactors or components
* Be versioned, scored, and comparable


### 🧱 Scope of Work

#### 4.1 **Feature Engineering Pipeline**

**Goal:** Build a deterministic, schema-validated pipeline to convert raw sensor data into a standardized ML feature format.

**Features:**

* Accepts a pandas DataFrame or streaming records
* Computes:
  * Rolling window statistics (mean, std, min, max)
  * First-order deltas between consecutive readings
  * Time-based features (e.g., cycles since start or last fault)
  * Component-normalized indicators (e.g., scaled pressure drop)
* Returns a feature matrix and target values (`true_RUL`)

**Interface:**

```python
def generate_features(df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame
```

**Deliverables:**

* `models/feature_engineering.py`
* Feature generation script for batch ML prep
* Test suite with deterministic outputs


**Dependencies:** `pandas`, `numpy`, optional `tsfresh` or `scipy`**Estimated Time:** 4 hours


#### 4.2 **Deep Learning Model for RUL Estimation**

**Goal:** Implement a PyTorch-based model (initially LSTM) to predict the number of cycles remaining before component failure.

**Model Characteristics:**

* Input: sliding windows of time-series features (e.g., 30 cycles x 6 features)
* Output: scalar RUL prediction
* Supports masking or padding for incomplete sequences
* Can be easily extended to GRU, Transformer, or BiLSTM

**Interface:**

```python
class RULLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int)
    def forward(self, x: Tensor) -> Tensor
```

**Deliverables:**

* `models/rul_model.py`
* Train/test scripts using MLflow logging
* Plotting and evaluation scripts (loss, predicted vs actual)

**Training Includes:**

* Early stopping
* Learning rate scheduler
* Experiment parameterization via config file


**Dependencies:** `torch`, `mlflow`, `matplotlib`**Estimated Time:** 6 hours


#### 4.3 **Training Workflow with MLflow Integration**

**Goal:** Implement a structured training script that uses generated features, trains an RUL model, evaluates it, and logs the full experiment with MLflow.

**Workflow:**

* Load dataset (from Parquet or CSV)
* Generate or load features
* Train model (with optional validation split)
* Evaluate with metrics: MAE, RMSE, R²
* Log:
  * Parameters
  * Final metrics
  * Model artifact
  * Optional SHAP plots

**Interface:**

```bash
python train.py --config config.yaml
```

**Deliverables:**

* `ml_pipeline/train.py`
* Model training curve plots
* Logs in MLflow under experiment name (e.g., "RUL_LSTM_2025_Q1")

**Config Sample:**

```yaml
model_type: "LSTM"
hidden_dim: 64
epochs: 40
batch_size: 32
learning_rate: 0.001
sequence_length: 30
```


**Dependencies:** `PyYAML`, `mlflow`, `torch`, `sklearn`, `shap`**Estimated Time:** 4 hours


#### 4.4 **Model Validation Script**

**Goal:** Evaluate any saved model from MLflow on new test data, producing metrics and diagnostics.

**Features:**

* Loads model using MLflow `model_uri`
* Accepts test data file (Parquet/CSV)
* Produces:
  * MAE, RMSE, R²
  * Residual plot
  * Time series plot of RUL (actual vs predicted)
  * SHAP feature importance

**Deliverables:**

* `ml_pipeline/validate.py`
* MLflow-compatible logging
* Reusable functions for evaluation


**Dependencies:** `mlflow`, `matplotlib`, `sklearn`, `shap`**Estimated Time:** 3 hours


#### 4.5 **Anomaly Detection Module**

**Goal:** Implement an unsupervised anomaly detector using a basic statistical method (e.g., Isolation Forest or PCA) to flag "out-of-pattern" sensor behaviors.

**Model Types:**

* **Isolation Forest**: Outlier score based on sub-sampling
* **PCA + Mahalanobis Distance**: Distance from projection plane
* Optional: Local Outlier Factor (for comparative testing)

**Features:**

* Model trains on normal cycles only
* Output: anomaly score (continuous) or binary flag
* Supports threshold tuning

**Interface:**

```python
class AnomalyDetector:
    def fit(self, X: pd.DataFrame)
    def predict(self, X: pd.DataFrame) -> np.ndarray
```

**Deliverables:**

* `models/anomaly_detector.py`
* Offline training and evaluation script
* Score distributions, ROC/PR curve


**Dependencies:** `sklearn`, `scipy`, `matplotlib`**Estimated Time:** 3 hours


#### 4.6 **Hyperparameter Tuning and Documentation**

**Goal:** Run grid search or random search over core model hyperparameters and document best configurations.

**Hyperparams to test:**

* LSTM: `hidden_dim`, `sequence_length`, `learning_rate`, `batch_size`
* Anomaly: contamination rate, # of components (PCA)

**Deliverables:**

* Updated `config.yaml`
* Training logs in MLflow
* Summary doc in `docs/model_experiments.md`


**Dependencies:** `mlflow`, `json`, `numpy`, `sklearn`**Estimated Time:** 3 hours


### 📈 Metrics of Success

| Target | Outcome |
|----|----|
| Feature pipeline | Deterministic, complete, unit-tested |
| RUL model | Trained, logged in MLflow with RMSE < benchmark |
| Anomaly detector | Produces valid outlier scores, logs ROC curve |
| Validation script | Loads any model and evaluates against new data |
| MLflow logging | Includes params, metrics, artifacts, config hash |


### ⏱️ Time Estimate Summary

| Task | Est. Hours |
|----|----|
| Feature Engineering | 4 hrs |
| RUL Model (PyTorch) | 6 hrs |
| Training Workflow + MLflow | 4 hrs |
| Model Validation | 3 hrs |
| Anomaly Detection | 3 hrs |
| Hyperparameter Search | 3 hrs |
| **Total** | **23 hours** |


### 🔄 Dependencies & Downstream Integrations

| Dependency | Consumer |
|----|----|
| ✅ `feature_engineering.py` | Used in `train.py`, `validate.py`, and eventually gRPC |
| ✅ `mlflow_registry.py` | Called by `train.py`, `grpc_server.py` |
| ✅ `parquet_writer.py` | Provides input to model training |
| ✅ `grpc_server.py` | Uses trained model for inference |
| ✅ `audit_logger.py` | Uses `model_uri` from validation output |


### 📌 Out of Scope

* Real-time inference (handled in Phase 5)
* Model deployment (handled in Phase 5 & 8)
* Retraining logic or continual learning (future)
* Federated learning or site-specific fine-tuning (future roadmap)


### 🧪 Test Plan

| Component | Test Case | Expected Output |
|----|----|----|
| Feature gen | 30-cycle input window | Returns shape (1, 30, N_features) |
| RUL model | Predict on test set | RMSE computed and logged |
| Anomaly model | Score test set | Outlier scores distributed correctly |
| MLflow log | Save model + metrics | Viewable in MLflow UI |
| Validation | Load + eval any model | Outputs metrics, plots, SHAP explanation |

# 📝 Product Requirements Document (PRD)

## Phase 5: Inference Pipeline & gRPC Serving



### 🎯 Objective

To implement a **high-performance, production-style ML inference pipeline** that serves Remaining Useful Life (RUL) predictions via a structured and versioned **gRPC API**, using models trained and registered in Phase 4.

This phase establishes the "runtime" side of the ML pipeline:

* The model is loaded from **MLflow**
* Predictions are served on structured input using **gRPC**
* All predictions are **audited and version-tracked**
* The service is **containerized** and deployed via **Cloud Run**

This phase is also a key integration point:

* Streaming data from Kafka consumers is now **used in real-time inference**
* gRPC service is callable from both streaming agents and diagnostic dashboards
* It lays the foundation for edge-based inference (one container per site)


### 🔍 Background

In real SMR deployments, inference pipelines will need to:

* Operate **close to the edge** (e.g., on-site or reactor-local)
* Serve predictions at **low latency**
* Be **strictly typed**, **auditable**, and **deterministically versioned**
* Eventually support **multi-component, multi-model orchestration**

This phase models that production reality using:

* **gRPC** (vs. REST) for performance and schema enforcement
* **MLflow** as the model registry
* **Protobuf schemas** to define input/output contract
* **Audit hooks** to ensure every prediction is logged with traceability

This phase also includes the **gRPC client integration into the Kafka consumer**, creating the first end-to-end loop: `simulation → streaming → inference → audit`.


### 🧱 Scope of Work

#### 5.1 **Protobuf Definition for gRPC API**

**Goal:** Define the gRPC service contract for submitting time-windowed telemetry features and receiving predicted RUL with version tracking.

**File:** `ml_pipeline/rul.proto`

**Schema:**

```proto
syntax = "proto3";

message SensorWindow {
  string site_id = 1;
  string reactor_id = 2;
  string component = 3;
  repeated float features = 4;
  string timestamp = 5;
}

message PredictRULRequest {
  SensorWindow window = 1;
}

message PredictRULResponse {
  float predicted_rul = 1;
  string model_version = 2;
  string timestamp = 3;
}

service RULInferenceService {
  rpc PredictRUL(PredictRULRequest) returns (PredictRULResponse);
}
```

**Deliverables:**

* `rul.proto` file in `ml_pipeline/`
* Compiled Python classes via `grpcio-tools`
* Documented schema version and example message

**Estimated Time:** 1 hour


#### 5.2 **gRPC Inference Server**

**Goal:** Build a robust, concurrent, production-ready inference server in Python using `grpc.aio`, powered by models pulled from MLflow.

**Features:**

* Loads model from MLflow using `model_uri`
* Validates input schema (feature count, data type)
* Runs RUL prediction on structured input
* Logs every prediction to audit module with:
  * Model version
  * Input features
  * Output
  * Site/component/timestamp
* Handles multiple concurrent requests (async)
* Reloads model cleanly if version changes

**File:** `ml_pipeline/grpc_server.py`

**Class Interface:**

```python
class RULInferenceService(RULServicer):
    async def PredictRUL(self, request, context):
        # Validate
        # Run model
        # Return response
```

**Deliverables:**

* gRPC server executable
* Test harness (curl or client script)
* Integration with `audit_logger.py`
* Integration with `mlflow_registry.py`


**Dependencies:** `grpcio`, `grpcio-tools`, `mlflow`, `pydantic`, `torch`, `protobuf`**Estimated Time:** 6 hours


#### 5.3 **gRPC Client for Streaming Integration**

**Goal:** Integrate the gRPC inference server into the Kafka consumer pipeline, so that streaming telemetry results in live RUL predictions.

**Features:**

* Extracts 30-cycle window from `kafka_consumer.py`
* Converts windowed features into protobuf message
* Sends request to gRPC inference server
* Receives `predicted_rul` + `model_version`
* Pushes result to:
  * `influx_writer.py` (for dashboarding)
  * `audit_logger.py` (for compliance)
  * `alerts` topic (if RUL is critically low)

**File:** `ml_pipeline/grpc_client.py`

**Deliverables:**

* Client utility with retry logic and logging
* Embedded in Kafka consumer
* Handles batching and serialization

**Estimated Time:** 3 hours


#### 5.4 **Optional Model Export to TorchScript**

**Goal:** Export model to TorchScript for fast, standalone inference. This reduces PyTorch dependency and speeds up serving.

**Use Cases:**

* Deployed inference containers
* Edge inferencing
* Interoperability with other languages

**File:** `ml_pipeline/export_model.py`

**Deliverables:**

* Script to export model to `.pt`
* Load + inference test
* Benchmark vs standard PyTorch loading

**Estimated Time:** 2 hours


#### 5.5 **Containerization and Cloud Deployment**

**Goal:** Package the gRPC server into a Docker container and deploy it to GCP Cloud Run. Test inference from remote client.

**Features:**

* Container image with PyTorch + MLflow + gRPC
* MLflow connection via env var
* gRPC port exposed (default 50051)
* Cloud Run deployment via `gcp_setup.sh`
* Example deployment logs + test output

**Deliverables:**

* Dockerfile for gRPC server
* `gcp_setup.sh` for Cloud Run deployment
* Cloud Run URL or IP with TLS config
* Test script with real prediction


**Dependencies:** Docker, GCP CLI, Cloud Run**Estimated Time:** 3 hours


### 📈 Metrics of Success

| Target | Outcome |
|----|----|
| gRPC service | Accepts requests and returns RUL prediction with version |
| MLflow | Models are pulled from registry, not hardcoded |
| Streaming pipeline | Live streaming → gRPC inference loop is functional |
| Auditing | Every prediction is logged with full context |
| Cloud deployment | Inference server is callable from remote script |


### ⏱️ Time Estimate Summary

| Task | Description | Est. Hours |
|----|----|----|
| 5.1 | Define protobuf schema | 1 hr |
| 5.2 | Build gRPC inference server | 6 hrs |
| 5.3 | Add gRPC client to streaming | 3 hrs |
| 5.4 | Export to TorchScript | 2 hrs |
| 5.5 | Containerize & deploy to Cloud Run | 3 hrs |
| **Total** | — | **15 hours** |


### 🔄 Dependencies & Downstream Integrations

| Dependency | Consumer |
|----|----|
| ✅ `models/rul_model.py` | Used for TorchScript export or MLflow model |
| ✅ `mlflow_registry.py` | Loads model artifact for gRPC inference |
| ✅ `audit_logger.py` | Logs every prediction |
| ✅ `kafka_consumer.py` | Sends requests to gRPC via client |
| ✅ `influx_writer.py` | Stores predictions for dashboard display |


### 📌 Out of Scope

* Batch inference (handled in `validate.py`)
* Multi-model serving (future: different components/models per type)
* REST API fallback (not part of current architecture)
* Federated or on-device inference (future roadmap)


### 🧪 Test Plan

| Component | Test Case | Expected Result |
|----|----|----|
| gRPC server | Request with 30-cycle window | Returns float RUL and model version |
| gRPC client | Send request from Kafka consumer | Response received within <300ms |
| Audit logging | One log per prediction | Contains input, output, version, timestamp |
| Cloud Run | Deploy container and hit endpoint | Logs prediction remotely |
| Model loading | MLflow URI → model object | Correct artifact loaded, no errors |

# 📝 Product Requirements Document (PRD)

## Phase 6: Dashboards & Visualization Layer



### 🎯 Objective

To create a structured, real-time visualization layer for both **raw sensor telemetry** and **ML predictions** by building:



1. **Grafana dashboards** for site- and reactor-level monitoring using data from InfluxDB.
2. An optional **Gradio interface** for visualizing ML model predictions interactively during demos or experiments.
3. A foundation for **RUL trend tracking, threshold alerting, and anomaly visualization**.
4. User-friendly selection of components by `site_id`, `reactor_id`, and `component`, aligned with the system’s multi-deployment design.

This visualization layer provides the **first true human interface** into the system, allowing stakeholders—engineers, operators, and potential collaborators—to **observe system health**, validate ML model behavior, and assess predictive maintenance utility in near-real time.


### 🔍 Background

In applied ML systems—particularly in **industrial, safety-adjacent domains**—building dashboards is not just about display, but about **transparency, trust, and situational awareness**. Regulators, operators, engineers, and executives need to understand:

* What the system sees
* What the model is predicting
* When things are about to break
* How reliable the prediction is

Grafana is the natural choice due to its:

* Native integration with **InfluxDB**
* Strong support for **templating, alerts, and annotation**
* Familiarity in operational engineering teams

**Gradio** may optionally be included to provide:

* A lightweight demo interface for **interactive ML model exploration**
* A visual wrapper for the gRPC API during presentations or experiments

This phase is not about full product UX—it’s about **building the observability layer** for ML-in-production in a critical environment.


### 🧱 Scope of Work

#### 6.1 **Configure InfluxDB as Grafana Data Source**

**Goal:** Enable Grafana to query sensor and prediction data stored in InfluxDB, using appropriate tag hierarchies and filters.

**Tasks:**

* Connect Grafana to InfluxDB instance (local Docker or Cloud-hosted)
* Verify metric ingestion (from `influx_logger.py`)
* Configure measurement names, retention policies, and tag filters

**Tags to index:**

* `site_id`
* `reactor_id`
* `component`
* `metric_type` (`sensor`, `prediction`, `alert`)

**Deliverables:**

* Grafana data source config
* Sample test query via GUI and CLI
* Grafana login + dashboard access documentation


**Dependencies:** `InfluxDB`, `Docker`, Grafana image**Estimated Time:** 2 hours


#### 6.2 **Design and Build Core Grafana Dashboards**

**Goal:** Create modular, templated dashboards for visualizing:

* Real-time sensor data (from InfluxDB)
* Predicted RUL (from ML model)
* Component status (degradation, fault conditions)

**Dashboard Features:**

* Component-level panel per signal (`temperature_in`, `pressure_drop`, etc.)
* RUL time-series graph
* Combined “health index” panel (based on performance metrics)
* Templated variables:
  * `site_id` (dropdown)
  * `reactor_id` (dependent dropdown)
  * `component` (checkbox selector)

**Dashboards to build:**

* `Reactor Overview`: All components at a single site/reactor
* `Component Detail`: All metrics and RUL for a single component
* `Fleet Health Summary`: Small multiples across sites/reactors

**Deliverables:**

* `dashboard/grafana_config.json`: Prebuilt dashboards as exportable JSON
* Folder structure in Grafana UI for multi-site support
* Color-coded RUL alert lines (e.g., RUL < 25 → orange; RUL < 10 → red)


**Dependencies:** `Grafana`, Influx query builder (Flux/SQL), JSON dashboard config**Estimated Time:** 4 hours


#### 6.3 **Configure Alert Rules in Grafana**

**Goal:** Trigger visible alerts or annotations in Grafana when:

* Model predicts low RUL (e.g., < 20)
* Sudden performance drop or sensor anomaly is detected
* Data stops flowing from a given site/reactor

**Features:**

* Alert rules per component (RUL, pressure drop, etc.)
* Email or log alerts (optional for this demo)
* Visual annotations (on time-series charts)
* Threshold bands on prediction panels

**Deliverables:**

* Alert rule config in Grafana
* Notification channels (placeholder for now)
* Visual annotation panel for live faults


**Dependencies:** Grafana Alerts, optionally Alertmanager**Estimated Time:** 2 hours


#### 6.4 **Optional: Interactive ML Viewer with Gradio**

**Goal:** Provide an interactive demo interface to explore ML predictions and model behavior, powered by the gRPC client.

**Use Cases:**

* Run ad hoc RUL predictions from sample data
* View SHAP importance or inference metadata
* Compare predictions across feature variations

**Features:**

* Upload telemetry CSV or select simulated component
* Form inputs for sensor metrics (manually adjustable)
* Button to trigger prediction via gRPC
* Output: RUL, model version, optional SHAP bar chart

**Interface:**

```python
def launch_gradio_demo():
    # Input fields
    # gRPC request wrapper
    # Display predicted RUL + explanation
```

**Deliverables:**

* `dashboard/gradio_app.py`
* Documentation for launching locally or via public link
* Optional integration with prediction logger


**Dependencies:** `gradio`, `grpcio`, `shap`**Estimated Time:** 3 hours


### 📈 Metrics of Success

| Target | Expected Outcome |
|----|----|
| Grafana data source | InfluxDB metrics accessible, browsable |
| Reactor Overview dashboard | Live metrics + predictions, templated filters |
| Alert configuration | Visual annotations on threshold breach |
| Gradio app | Working demo interface for single-sample predictions |
| Docs | Screenshot-driven guide for dashboard use |


### ⏱️ Time Estimate Summary

| Task | Description | Est. Hours |
|----|----|----|
| 6.1 | InfluxDB integration | 2 hrs |
| 6.2 | Build Grafana dashboards | 4 hrs |
| 6.3 | Add alerting rules | 2 hrs |
| 6.4 | Gradio demo UI (optional) | 3 hrs |
| **Total** | — | **11 hours** |


### 🔄 Dependencies & Downstream Integrations

| Upstream | Used For |
|----|----|
| ✅ `influx_logger.py` | Supplies telemetry + prediction metrics |
| ✅ `grpc_client.py` | Enables Gradio to call inference API |
| ✅ `audit_logger.py` | Used to cross-check predictions in dashboard |


### 📌 Out of Scope

* User authentication/authorization
* Frontend product-style UI
* Embedded dashboard in external applications
* Automated Grafana provisioning (manual JSON import for now)


### 🧪 Test Plan

| Component | Test Case | Expected Output |
|----|----|----|
| Grafana | Open dashboard | Sensor values plotted live |
| RUL panel | Receive prediction | Line plot updates, color threshold triggers |
| Anomaly alert | Inject fault via simulator | Alert fires and annotates chart |
| Gradio app | Input sensor vector | gRPC returns RUL + model version |
| Data sync | Disable simulator | Grafana “no data” alerts appear |

# 📝 Product Requirements Document (PRD)

## Phase 7: Audit & Compliance Layer



### 🎯 Objective

To build a robust, modular audit layer that ensures **every ML prediction is fully traceable**: what was predicted, when, on what input, using which model. This phase is **crucial for demonstrating regulatory readiness**, even if formal audits are not required at the current company stage.

This includes:



1. A **prediction logging system** that captures input, output, metadata, and model version.
2. Integration of audit logging into the **gRPC inference pipeline** (Phase 5).
3. Use of **structured, queryable formats** for storage (Parquet, SQLite, or GCP Cloud SQL).
4. Optional hooks for **schema validation**, **version control**, and **prediction replay**.
5. Documentation referencing **NRC-relevant standards** (e.g., traceability within 10 CFR Part 50/52 pre-licensing context).


### 🔍 Background

The eventual deployment of ML in nuclear settings—even for non-safety-critical systems like predictive maintenance—will require:

* **Complete traceability** of inference decisions
* **Version control** of models and inputs
* **Evidence of reproducibility** for any automated recommendation
* **Audit trail alignment** with digital twin verification/validation (future V&V under ASME V&V 20 or similar)

This phase demonstrates your understanding of the **long-tail lifecycle of ML in industry**—and prepares the company to integrate with regulatory workflows without being caught unprepared later.


### 🧱 Scope of Work

#### 7.1 **Prediction Audit Logger**

**Goal:** Create a structured logging system that captures the full context of each ML prediction.

**Logged Fields:**

* `timestamp` (UTC)
* `site_id`, `reactor_id`, `component`
* `model_name`, `model_version`
* `input_features` (optional: hash or summary)
* `predicted_rul`
* `inference_time_ms`

**Interface:**

```python
class AuditLogger:
    def log_prediction(
        site_id: str,
        reactor_id: str,
        component: str,
        model_version: str,
        input_features: List[float],
        predicted_rul: float,
        timestamp: str
    )
```

**Storage Targets:**

* Default: `audit/audit_log.parquet`
* Optional: SQLite (for queries)
* Future: BigQuery or Cloud SQL

**Deliverables:**

* `audit/prediction_logger.py`
* Unit tests for logger
* Configurable log rotation/append mode


**Dependencies:** `pandas`, `pyarrow`, `datetime`, `uuid`, `os`**Estimated Time:** 3 hours


#### 7.2 **gRPC Inference Integration**

**Goal:** Ensure every call to the gRPC `PredictRUL()` endpoint logs a complete record using the audit logger.

**Features:**

* Injected logger object into gRPC server context
* Logs immediately after successful prediction
* Catches logging failures without interrupting inference

**Deliverables:**

* Integration in `ml_pipeline/grpc_server.py`
* Logging test case
* Inference audit trail demo with CLI or notebook


**Dependencies:** `audit_logger`, `grpcio`**Estimated Time:** 2 hours


#### 7.3 **Compliance Hooks and Enforcement (Soft Gate)**

**Goal:** Introduce soft gate-keeping hooks that prevent serving from:

* **Unregistered model versions**
* **Invalid feature sets**

These checks are useful even today as a way to *instill discipline in serving*.

**Hook Examples:**

* Validate input feature count and schema against hash or config
* Load model version from MLflow and cross-check signature
* Optionally: store signed config object per model

**Deliverables:**

* `audit/compliance_hooks.py`
* Functions:
  * `validate_feature_schema()`
  * `verify_model_registration(model_version)`
* Test cases for failure modes


**Dependencies:** `mlflow`, `pydantic`, `hashlib`**Estimated Time:** 3 hours


#### 7.4 **Documentation: Regulatory Readiness & Traceability**

**Goal:** Create a written explainer for internal stakeholders and external reviewers that documents how the audit layer prepares the company for future compliance requirements.

**Contents:**

* Summary of logging system
* References to NRC-related traceability principles
* Mention of ASME V&V 20 (model validation)
* How audit logs connect to licensing or QA workflows
* Examples of trace replay: from a given timestamp, reconstruct the model output

**Deliverables:**

* `audit/compliance_overview.md`
* 1–2 diagrams showing audit data flow
* Example audit log and query

**Estimated Time:** 2 hours


### 📈 Metrics of Success

| Target | Expected Outcome |
|----|----|
| Logging | Every inference is logged with full metadata |
| Schema validation | Prediction with malformed input is flagged |
| Model version control | Logs match MLflow versions |
| Replayability | Can reproduce a prediction from audit log |
| Documentation | Written guide is clear, actionable, and strategic |


### ⏱️ Time Estimate Summary

| Task | Description | Est. Hours |
|----|----|----|
| 7.1 | Prediction logger | 3 hrs |
| 7.2 | gRPC integration | 2 hrs |
| 7.3 | Compliance hooks | 3 hrs |
| 7.4 | Documentation + diagrams | 2 hrs |
| **Total** | — | **10 hours** |


### 🔄 Dependencies & Downstream Integrations

| Source | Dependency |
|----|----|
| ✅ `grpc_server.py` | AuditLogger is called after each prediction |
| ✅ `mlflow_registry.py` | Verifies model version and input hash |
| ✅ `grpc_client.py` | Supports re-run or prediction replay |
| ✅ `influx_writer.py` | Can cross-reference prediction logs for Grafana display |


### 📌 Out of Scope

* Automatic compliance with 10 CFR Part 50/52 (handled by QA/RA teams)
* Digital signature or cryptographic sealing of logs (future)
* Continuous validation against ground truth (future—requires real failure data)
* Real regulatory submission formatting (PDFs, XML, etc.)


### 🧪 Test Plan

| Component | Test Case | Expected Outcome |
|----|----|----|
| Audit logger | Log prediction with dummy data | Record saved to Parquet file |
| gRPC server | Trigger live prediction | Audit entry appears with full metadata |
| Compliance hook | Send malformed feature set | Validation fails, prediction blocked |
| Model check | Use unregistered version | Warning or exception raised |
| Documentation | Reviewed by peer | Meets clarity and domain-awareness goals |

# 📝 Product Requirements Document (PRD)

## Phase 8 – Infrastructure & Cloud Deployment



### 🎯 Objective

To implement the system infrastructure and deployment tooling required to make the entire `pm-stack-smr` project **environment-agnostic, containerized, portable, and cloud-deployable**.

This includes:



1. Defining containerization strategies using **Docker** to ensure each system component (Kafka, MLflow, InfluxDB, gRPC inference server, Grafana) can be started and run reliably in local and cloud environments.
2. Writing startup orchestration scripts using **Docker Compose** for local development and **GCP Cloud Run** (and optionally GKE) for cloud deployment.
3. Providing a reproducible environment for developers and stakeholders to spin up the complete stack with **minimal manual configuration**.
4. Capturing **infrastructure configuration documentation**, with step-by-step deployment guides and configuration references.
5. Preparing the system for **future CI/CD workflows**, including model redeployment triggers and cloud-based logging pipelines.

This phase is focused on **developer experience, operational integrity, and production foresight**—providing the environment scaffolding that future ML, data, and DevOps teams can adopt, modify, and scale.


### 🔍 Background

The goal of this system is not to remain in an experimental state—it is to **prototype what a future production ML/data stack will look like in the context of a nuclear startup**. That means:

* Models cannot live only in notebooks.
* Inference cannot depend on one machine or a single dev setup.
* Data systems must be orchestrated, observable, and documented.
* Audit and compliance workflows must persist across environments.
* Simulators, loggers, inference, and dashboards must all **coexist** in a reproducible and deployable state.

By containerizing the system and deploying it to the cloud, we make it both **real and demonstrable**—even with synthetic data.

This phase demonstrates to stakeholders that:

* You know how to deploy and operationalize machine learning systems.
* You think about **data/ML in the context of real infra** constraints.
* You're ready to lead a team that might one day **deploy edge inferencing at a reactor site**, not just run experiments on a laptop.


### 🧱 Scope of Work

#### 8.1 **Containerization of System Components (Dockerfiles)**

The first task is to ensure that each major component of the system can be run inside a reproducible, isolated, and portable Docker container.

This includes:

* The **gRPC inference server**, built on Python, gRPC, and PyTorch
* The **MLflow tracking server**, with local storage for now
* **InfluxDB**, configured with data retention and write authentication
* **Grafana**, with JSON-importable dashboards
* Optional: Kafka + Zookeeper (if not already containerized in Phase 2)

Each container must:

* Be tagged and versioned appropriately (e.g., `inference-server:0.1.0`)
* Contain only the necessary runtime dependencies
* Mount appropriate volumes for persistent storage (e.g., MLflow logs, audit logs)
* Accept configuration via `ENV` or `.env` files
* Be security-conscious (non-root user, minimal base image)

Deliverables:

* `infra/Dockerfile.grpc_inference`
* `infra/Dockerfile.mlflow`
* `infra/Dockerfile.simulation` (optional)
* Documentation on how to build and run each component individually

This ensures that even without orchestration, every critical service is **packaged, portable, and isolated**.


**Estimated Time:** 3 hours**Dependencies:** Docker, existing code modules


#### 8.2 **Local Development Orchestration with Docker Compose**

Once individual containers are defined, they will be orchestrated locally using Docker Compose. This allows the entire system—telemetry simulators, Kafka, InfluxDB, inference server, MLflow, Grafana—to be spun up with a single command.

This Docker Compose configuration must:

* Define shared networks between services
* Provide health checks and restart policies
* Handle dependency ordering (e.g., MLflow waits for storage volume, Grafana waits for InfluxDB)
* Mount source code into containers for rapid iteration
* Define named volumes for persistent audit and time-series data

Deliverables:

* `infra/docker-compose.yml`: core file defining services
* `infra/docker-compose.override.yml`: optional overrides for local-only behavior
* README instructions with sample `docker-compose up` output and verification steps

This is the starting point for all devs who need to test or extend the system.


**Estimated Time:** 3 hours**Dependencies:** Docker, docker-compose, Phase 1–7 components


#### 8.3 **Cloud Deployment via GCP Cloud Run (gRPC Server)**

To make the system deployable in a real environment—and to demonstrate deployment readiness—the gRPC inference service will be deployed to **Google Cloud Run**, Google’s fully managed container runtime.

This cloud deployment must:

* Accept a built container (`gcr.io/<project-id>/inference-server`)
* Pull the most recent model from MLflow (hosted locally or also in the cloud)
* Accept and respond to remote gRPC clients
* Use service accounts and permissions configured via GCP IAM
* Be auto-scalable (0–N instances)
* Expose a public (or private) endpoint for use in the Kafka consumer or external dashboard

Deliverables:

* `infra/gcp_setup.sh`: CLI script that deploys the container to Cloud Run
* Optional: `infra/cloudbuild.yaml` for CI-compatible builds
* `infra/deploy_config.yaml`: configs for endpoint, port, authentication

Cloud Run will be the **first cloud-hosted component of the system**, proving that the ML inference capability is cloud-native and infrastructure-aware.


**Estimated Time:** 3 hours**Dependencies:** GCP CLI, Docker registry, MLflow registry


#### 8.4 **Cloud Storage for Audit Logs and Artifacts (Optional)**

To enable future compliance audits, model tracking, and data archiving, support for cloud-based storage will be configured.

This includes:

* Backing up `audit_log.parquet` to a GCS bucket
* Optionally storing MLflow artifacts (models, metrics, plots) in GCS
* Writing documentation for future integrations with BigQuery or Cloud SQL

Deliverables:

* Environment variable injection for storage paths
* `infra/storage_config.sh` script for initial bucket setup
* Security policies for public/private access

This ensures the system is **production-compatible** and **compliance-forward**, even in demo form.


**Estimated Time:** 2 hours**Dependencies:** GCP SDK, audit module, MLflow artifact logging


#### 8.5 **Infrastructure Documentation and Setup Instructions**

A project’s infrastructure is only useful if it is understandable and reproducible. This deliverable provides a **step-by-step guide** to launching the full system—locally and in the cloud.

Contents of `infra/README.md`:

* How to run locally (`docker-compose`)
* How to build and test each image
* How to deploy to Cloud Run
* How to monitor service health
* How to connect to InfluxDB, MLflow, Grafana
* Environment variable templates
* Diagrams showing service-to-service architecture

This documentation is what makes your infra work **transferable** to teammates, future hires, and collaborators. It is also essential if you pitch your system in a walkthrough meeting.


**Estimated Time:** 2 hours**Dependencies:** All infra files above, diagram tools (Excalidraw, draw.io, etc.)


### 📈 Metrics of Success

| Target | Outcome |
|----|----|
| Local system | `docker-compose up` brings full stack online |
| gRPC on Cloud Run | Endpoint responds to requests from local client |
| Audit logs | Persist locally and are optionally uploaded to GCS |
| MLflow | Accessible from browser with artifact history |
| Documentation | Clear enough for another dev to reproduce full setup |


### ⏱️ Time Estimate Summary

| Task | Description | Est. Hours |
|----|----|----|
| 8.1 | Containerization | 3 hrs |
| 8.2 | Docker Compose orchestration | 3 hrs |
| 8.3 | Cloud Run deployment | 3 hrs |
| 8.4 | Cloud storage for audit/model logs | 2 hrs |
| 8.5 | Infrastructure documentation | 2 hrs |
| **Total** | — | **13 hours** |


### 🔄 Dependencies & Downstream Integrations

| Upstream | Consumes |
|----|----|
| ✅ `grpc_server.py` | Deployed as Cloud Run container |
| ✅ `audit_logger.py` | Writes logs to volume or GCS |
| ✅ `mlflow_registry.py` | Pulls from GCS or local artifact store |
| ✅ `influx_logger.py` | Writes to mounted InfluxDB volume (local) |


### 📌 Out of Scope

* CI/CD pipelines for automated model deployment (future)
* Infrastructure-as-code (Terraform, Pulumi—future enhancement)
* Kubernetes deployment (handled later with GKE if necessary)
* On-prem / air-gapped deployment (future, once hardware is introduced)


### 🧪 Test Plan

| Component | Test Case | Expected Output |
|----|----|----|
| Local Docker | `docker-compose up` | All containers launch, networked |
| Cloud Run | Call inference endpoint | RUL returned in <500ms |
| MLflow | UI loads, models are visible | Artifact, versioning available |
| Audit | File written locally | Log entries saved in audit file |
| Infra docs | Follow instructions on clean machine | Stack starts and functions end-to-end |

# 📝 Product Requirements Document (PRD)

## Phase 9: Documentation & Communication



### 🎯 Objective

To produce comprehensive, high-quality, and stakeholder-oriented documentation that **clearly communicates the design, purpose, functionality, and vision** of the predictive maintenance system for SMRs, including:



1. A **system overview document** that explains the full architecture and components of the project in depth.
2. A **high-resolution architecture diagram** that visually communicates system structure and data flow.
3. A **strategic CTO-facing write-up**, explaining how this system aligns with Applied Atomics’ business and regulatory future.
4. An optional, lightweight **slide deck** to structure verbal pitches or walkthrough demos.
5. Source-level inline documentation, config file examples, and a developer onboarding guide.
6. Clear documentation of **how to extend, deploy, and understand** the system—even for engineers who didn't build it.

This phase is about **transparency, polish, and positioning**—cementing your system (and yourself) as **engineering-leadership-caliber**.


### 🔍 Background

By this phase, the technical value of the system has been built. But that value is latent unless it is:

* **Visible** to collaborators, CTOs, and hiring stakeholders
* **Understandable** to engineers who didn’t build it
* **Navigable** to anyone needing to debug, extend, or present it
* **Persuasive** to someone deciding whether to hire you into a leadership role

Documentation here is **not an afterthought**—it is a **core asset**. The difference between a strong portfolio and a leadership pitch is often how well the work is **communicated**.

This phase creates the artifacts that can be used:

* In a live pitch
* In a follow-up reference email
* As a foundation for future internal adoption


### 🧱 Scope of Work

#### 9.1 **System Overview Document**

**Goal:** Provide a single, comprehensive Markdown file that describes the entire system—its architecture, components, motivation, technical choices, and future roadmap.

This document will be read by:

* CTO-level stakeholders
* Senior engineers vetting your work
* Potential future collaborators or team members

**Contents:**

* Project summary (goal, scope, context)
* Component breakdown:
  * Simulation
  * Streaming
  * ML pipeline
  * Serving + Inference
  * Audit + Compliance
  * Infrastructure + Cloud
* Explanation of each phase, including:
  * What it is
  * Why it exists
  * What it enables
* Reference to important files, configuration formats
* Links to key folders or functions

**File:** `docs/system_overview.md`

**Deliverables:**

* Markdown file with heading structure and embedded diagrams
* Code block snippets for illustration
* Explanation of key design tradeoffs and constraints


**Estimated Time:** 3 hours**Dependencies:** Everything built in Phases 1–8


#### 9.2 **Architecture Diagram**

**Goal:** Produce a clear, professional, and presentation-ready visual that shows the **entire system architecture**—components, data flow, interfaces, and site-based context.

This is the **single most important visual** in the entire project.

**Features:**

* High-level box-and-arrow diagram
* Shows multi-reactor/multi-site deployment
* Streaming pipeline (Kafka)
* Storage layer (Influx, Parquet)
* MLflow and audit trace
* gRPC serving loop
* Optional GCP deployment path

**Style Notes:**

* Use color coding for functional layers (simulation, ingestion, ML, observability)
* Include all key components: adapters, model registry, dashboards
* Annotate connections (e.g., “30-cycle window”, “prediction request”, “audit log”)

**Tools:** Excalidraw, draw.io, Whimsical, Lucidchart (any preferred vector tool)

**File:** `docs/architecture_diagram.png` and `.svg` (for editability)

**Deliverables:**

* Diagram PNG/SVG in `docs/`
* Embedded in `system_overview.md`
* Suitable for presentation slides or printout


**Estimated Time:** 2 hours**Dependencies:** Knowledge of all system components


#### 9.3 **CTO-Facing Write-Up**

**Goal:** Deliver a concise, clearly-written explanation of:

* What you built
* Why it matters
* How it aligns with Applied Atomics’ future
* How you would lead this into production
* Why you’re the right person to do that

**Tone:** Strategic, executive-facing, confident but not sales-y.

**Format:** Markdown or PDF summary, written like a senior engineer explaining to a startup CTO.

**Contents:**

* One-paragraph overview of system value
* Description of how it addresses:
  * Predictive maintenance needs
  * Regulatory readiness
  * Multi-reactor deployment
* Description of your role: “What I built, how I thought about it, how I’d lead it forward”
* A brief “next steps if this were live” section

**File:** `docs/cto_readme.md`

**Deliverables:**

* Strategic write-up tailored to your job pitch
* Clear alignment with the company’s roadmap
* Optional delivery in PDF format for sharing


**Estimated Time:** 2 hours**Dependencies:** Everything—especially Phase 7 (Audit/Compliance)


#### 9.4 **Developer Onboarding + Extension Docs**

**Goal:** Write internal-facing documentation so that a future engineer could:

* Set up the system
* Run simulations
* Train or deploy models
* Understand where everything lives
* Modify it safely

**Structure:**

* `docs/dev_setup.md`: Installation, dependencies, startup steps
* `docs/extending_system.md`: How to add a new component or model
* `docs/env_config_template.env`: Example environment config
* `docs/usage_cheatsheet.md`: One-liners for train, infer, simulate, log, deploy

These docs ensure that your work is:

* Maintainable
* Reusable
* Scalable beyond you

**Deliverables:**

* 3–4 Markdown files
* Commented `.env` config with explanations
* Optional: sample `Makefile` or shell scripts for common tasks


**Estimated Time:** 2 hours**Dependencies:** Infra (Phase 8), Simulation, MLflow


#### 9.5 **Optional Slide Deck for Presentation Framing**

**Goal:** Create a **lightweight, 5-slide deck** that helps structure a walkthrough or pitch meeting.

This is **not a product pitch deck**—it is a **technical architecture aid**, focused on narrative clarity and system flow.

**Slides:**



1. The Problem: What SMRs will need in terms of ML/data infra
2. The System: What this stack does and how it works
3. Architecture Diagram: Full system view
4. What I Built: Your role, system components, technical thinking
5. What This Enables: Next steps and scale-up vision

**Tools:** Google Slides, PowerPoint, or Markdown-to-PDF export

**Deliverables:**

* `docs/pm-stack-smr-pitch-deck.pdf`
* Editable source file (optional)


**Estimated Time:** 3 hours (optional)**Dependencies:** Diagram, write-up, system knowledge


### 📈 Metrics of Success

| Target | Outcome |
|----|----|
| System overview | Complete document covering all components |
| Diagram | Clean, high-quality, reusable image |
| CTO doc | Compelling strategic alignment narrative |
| Dev docs | Clear enough for onboarding in under 1 hour |
| Deck | Presentation-ready slides for structure (if needed) |


### ⏱️ Time Estimate Summary

| Task | Description | Est. Hours |
|----|----|----|
| 9.1 | System overview doc | 3 hrs |
| 9.2 | Architecture diagram | 2 hrs |
| 9.3 | CTO write-up | 2 hrs |
| 9.4 | Developer/internal docs | 2 hrs |
| 9.5 | (Optional) Slide deck | 3 hrs |
| **Total** | — | **9–12 hours** |


### 🔄 Dependencies & Downstream Integrations

| Artifact | Used In |
|----|----|
| `system_overview.md` | Internal onboarding, public reference |
| `architecture_diagram.png` | Pitch decks, slides, demos |
| `cto_readme.md` | Strategic job pitch to CTO |
| `dev_setup.md` | Any future team onboarding |
| `Makefile/scripts` | Future CI/CD workflows |


### 📌 Out of Scope

* Full documentation website (e.g., with MkDocs or Docusaurus)
* Swagger or protobuf auto-generated API docs
* Compliance documentation templates (Phase 7 already framed them)


### 🧪 Test Plan

| Component | Test Case | Expected Output |
|----|----|----|
| Overview doc | Read by a peer | They can describe system with no prior exposure |
| Diagram | Shown in slide deck | CTO understands component layout |
| Dev docs | Run setup from scratch | System launches end-to-end |
| CTO write-up | Reviewed by non-ML engineer | Value and leadership pitch are clear |
| Deck | Used in Zoom pitch | Each slide supports the spoken narrative |


