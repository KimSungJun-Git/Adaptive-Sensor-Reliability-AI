*[Read in English](README.md)*

# Adaptive Sensor Reliability AI

**다중 센서 일관성 기반 실시간 신뢰도 추정 및 적응형 EKF**

LiDAR Odometry / IMU / Wheel Encoder의 측정 관계와 독립적인 **Shadow EKF**의
Innovation·NIS를 AI가 분석하여 **(센서 × 자유도)별 신뢰도**를 추정하고, 그 결과를
EKF measurement covariance에 **bounded inflation**으로 반영하여 Wheel Slip 등
센서 이상 상황에서도 안정적인 상태 추정을 유지합니다.

```text
LiDAR + IMU + Encoder
        ↓
 Sensor Residuals ──┐
        ↓           │
 Shadow EKF (fixed R) → Innovation / NIS
        ↓
 Reliability AI (ONNX MLP)   s_v_encoder, s_ω_encoder, s_v_lidar, s_ω_lidar, s_ω_imu
        ↓
 Asymmetric Smoothing + Recovery Gate
        ↓
 Bounded R Inflation:  R = R₀·[1 + α(1−s)],  R₀ ≤ R ≤ R₀(1+α)
        ↓
 Adaptive EKF  (또는 robot_localization + Covariance Injector)
```

핵심 설계 (자세한 근거는 [개발계획서](Adaptive_Sensor_Reliability_AI_개발계획서.md)):

- **DOF 분리형 Reliability** — Wheel Slip 시 `s_v_encoder`만 하락하고
  `s_ω_encoder`, `s_v_lidar`는 유지됩니다. 멀쩡한 정보를 버리지 않습니다.
- **Shadow EKF** — AI 입력용 Innovation/NIS는 고정 covariance 필터에서 계산해
  Adaptive EKF → AI 피드백 루프(reliability latching)를 차단합니다.
- **AI는 센서값을 수정하지 않습니다** — 오직 covariance만 조정합니다.

## 패키지

| 패키지 | 내용 |
|---|---|
| `asr_core` | ROS 무의존 코어: EKF, 2D point-to-line ICP(+degeneracy), feature, fault, smoothing, rule baselines, metrics |
| `asr_msgs` | `SensorReliability`, `ReliabilityArray`, `Innovation`, `FaultStatus`, `LidarQuality` |
| `asr_sensing` | encoder twist, ICP LiDAR odometry, Gazebo GT, driver, fault injector |
| `asr_shadow_ekf` | 고정 R Shadow EKF (Innovation/NIS 발행) |
| `asr_reliability_estimator` | feature sync + ONNX 추론 + smoothing + Adaptive EKF + **Covariance Injector** |
| `asr_bringup` | Gazebo(TurtleBot3 waffle_pi) launch, 데모, 대시보드 |
| `asr_evaluation` | rosbag 추출, 캘리브레이션, fault 데이터셋, 다중 seed 학습(+Autoencoder 비교), Fixed/Rule/RuleZ/AI 비교, 월드 holdout, 스무딩 하이퍼파라미터 스윕 |

## 빠른 시작 (시뮬레이션 데모)

요구사항: ROS2 Humble, Gazebo Classic 11, `ros-humble-turtlebot3-gazebo`,
`pip install onnxruntime`.

```bash
colcon build --symlink-install && source install/setup.bash
ros2 launch asr_bringup asr_demo.launch.py scenario:=slip
```

Wheel Slip 시나리오가 22초쯤 시작되면 대시보드에서 `encoder s_v`만 하락하고
Adaptive EKF 궤적이 Fixed EKF보다 GT에 가깝게 유지되는 것을 확인할 수 있습니다.
`scenario:=` `normal | slip | enc_drift | imu_bias | lidar_sector | combo | ...`

### 온라인 파이프라인 스모크 테스트

`asr_core`(12개) + `asr_sensing`(6개) + `asr_reliability_estimator`(7개, 신규:
온라인 시각동기화 `Series` 링버퍼) — 총 25개 유닛 테스트는 ROS·Gazebo를 전혀
건드리지 않으므로 노드 간 배선(토픽
이름, 메시지 형태, 조용히 멈추는 노드)이 깨지는 건 못 잡습니다.
`smoke_test_live`는 데모를 실제로 띄워서 reliability 발행 주기와 (센서×DOF)별
신뢰도가 해당 시나리오의 fault 구간에서 실제로 하락하는지까지 확인합니다 —
느리지만(시나리오당 ~1분) 배포 전 체크리스트에 넣을 만한 유일한 종단 검증입니다.
gzserver의 간헐적 시작 실패는 `record_runs`와 동일한 재시도로 처리합니다.

```bash
python3 -m asr_evaluation.smoke_test_live --scenario slip   # 하나만
python3 -m asr_evaluation.smoke_test_live --all             # 10개 시나리오 전부
```

## 실제 로봇 배포 — robot_localization Covariance Injector

자체 EKF를 강요하지 않습니다. Covariance Injector가 기존 센서 토픽을
**동일 메시지 타입 + reliability 반영 covariance**로 republish하므로,
사용자의 기존 `ekf_filter_node` 설정에서 토픽 한 줄만 바꾸면 됩니다:

```yaml
odom0: /odom/adaptive        # 기존: /odom
imu0: /imu/adaptive
twist0: /lidar_twist/adaptive
```

시뮬레이션에서 전체 체인 확인:

```bash
ros2 launch asr_bringup rl_demo.launch.py scenario:=slip
```

이 경로는 지금까지 라이브로 검증된 적이 없었는데(smoke_test_live는 자체 EKF
경로인 `asr_demo.launch.py`만 검증, Covariance Injector + 실제
`robot_localization/ekf_node`는 별도), 이번에 직접 띄워서 확인했습니다: 정상
기동(`ros-humble-robot-localization` 3.5.4), `/odom/adaptive`·`/asr/imu/adaptive`
·`/asr/lidar_twist/adaptive`가 원본과 동일 메시지 타입으로 정상 republish되고
`/odometry/filtered`가 유효한 pose를 발행합니다. 주입된 covariance 자체는
작고(`v_encoder` twist covariance ~1e-4~1e-3 수준) 정상 범위였고, slip fault
윈도우에 들어가면서 `ekf_filter_node`가 자체 보고하는 x/y 위치 공분산이 20초
사이 78→200(x분산)으로 커지는 것도 확인했습니다 — 처음엔 x-y 교차항이 x분산보다
커서 공분산 행렬이 깨진 줄 알았지만, 2x2 부분행렬 판별식(199.9×10038.3 −
1357.98² ≈ 162,854 > 0)을 확인해보니 정상적인 양의준정부호 행렬입니다(vx·vω만
융합하고 x,y는 융합 안 하는 구조라 상관계수가 ~0.96으로 높게 나오는 것뿐).
즉 "encoder를 못 믿게 되면 로봇이 자기 위치도 점점 더 모르게 된다"는 의도된
동작이 실제 `robot_localization` 스택에서도 수학적으로 올바르게 나타납니다.
다만 `asr_bringup/config/robot_localization.yaml`은 `process_noise_covariance`를
따로 튜닝하지 않고 robot_localization 기본값을 씁니다 — 실제 로봇에 배포할 때는
이 부분을 로봇의 실제 동역학에 맞게 조정하는 게 좋습니다(이건 이 프로젝트의
AI/신뢰도 시스템과 무관한, robot_localization 자체의 표준 튜닝 항목입니다).

## 학습/실험 재현 (Phase 1→10)

추가 요구사항: `pip install torch scikit-learn pandas`(MLP 학습·Isolation Forest
비교·CSV 집계에 필요; 위 "빠른 시작"의 onnxruntime과 달리 데모 실행에는 필요
없고 오직 이 학습 파이프라인에서만 씀 — `asr_evaluation/package.xml`엔
`python3-sklearn`/`python3-pandas`로 선언돼 있어 `rosdep install`로도 풀리지만,
torch는 rosdep으로 못 푸는 PyPI 전용 패키지라 pip로 따로 설치해야 함). 이 전부를
이미 갖춘 이미지가 아래 [Docker](#docker)의 `Dockerfile.dev`입니다.

```bash
source install/setup.bash
for w in tb3_world house stage4; do                                          # Phase 1: 3개 월드
  python3 -m asr_evaluation.record_runs --world $w --n 10 --duration 90 --out data/raw
done
python3 -m asr_evaluation.extract_bag --raw data/raw --out data/extracted   # Phase 2
python3 -m asr_evaluation.calibrate                                         # NOMINAL_STD 제안값 출력
python3 -m asr_evaluation.build_dataset                                     # Phase 4-6
asr_evaluation/run_experiments.sh                                           # Phase 7-10 전체 매트릭스
```

- `build_dataset`: run 단위·월드별 층화 분할(시계열 누수 방지), **23개 시나리오**
  (정상 / encoder v·ω bias·drift·noise·stuck·scale·spike·quant·slip / IMU
  bias·drift·noise·spike·dropout / LiDAR sector·noise·dropout / 복합 / **정지 상태
  전용 실험 2종**)를 기록된 정상 주행에 오프라인 주입. 정지 실험은 각 run의 ground
  truth에서 실제로 멈춘 구간을 찾아 그 구간에만 fault를 주입합니다(`find_stationary_window`).
- `train`: Rule(순간 잔차) · **RuleZ**(AI와 동일한 윈도우 feature의 z-score + NIS 게이팅,
  강한 규칙 baseline) · IsolationForest · **Autoencoder**(정상 데이터로만 학습, 재구성
  오차를 calibration해 이상 점수로 사용) · MLP × 3 seeds → 최적 seed를 ONNX로 배포.
  `--holdout <world>`로 leave-one-world-out
- `compare`: Fixed / Rule / RuleZ / AI(seed 평균±std) ATE·RPE, 탐지 F1·AUROC·지연·복구,
  오탐율(`far_clean`은 GT 기준으로 센서가 실제로 틀린 구간—충돌 등—을 제외),
  `sat_frac`(covariance가 R_max 근처에 머문 비율, 계획서 21.4절)
- `sweep_smoothing`: 재학습 없이(스무딩·covariance 파라미터는 ONNX 추론 이후 단계라)
  `GAMMA_FALL/RISE`, `RECOVERY_TICKS`, `ALPHA_INFLATION`, `S_REJECT`를 현재 기본값
  주변에서 좌표별로 스윕. 결과는 `asr_core/params.py`에 반영되어 있습니다.

## 결과 (3개 월드 × 32런 × 23시나리오, 월드별 층화 run 분리 + leave-one-world-out)

`tb3_world`(TurtleBot3 기본 아레나) / `house` / `stage4`(장애물 스테이지) 3개 Gazebo
월드에서 각 10~12런씩 수집. **AI는 3 seed 평균**으로 보고합니다(`results/main/summary.md`).
스무딩·covariance 하이퍼파라미터는 `sweep_smoothing`으로 튜닝한 값입니다
(`RECOVERY_TICKS` 10→5, `S_REJECT` 0.10→0.05 — 재학습 없이 정상 주행 ATE 65%,
fault ATE 16%, 오탐율 21% 동시 개선; `results/smoothing_sweep.json`).

### 시나리오별 ATE RMSE [m] — Fixed → Rule → RuleZ → **AI**

RuleZ는 AI와 동일한 윈도우 feature에 z-score + NIS 게이팅을 적용한 강한 규칙
baseline입니다(단순 순간 threshold인 Rule보다 훨씬 셉니다).

| 시나리오 | Fixed | Rule | RuleZ | **AI (±seed std)** |
|---|---:|---:|---:|---:|
| normal | 0.012 | 0.012 | 0.011 | **0.032 ± 0.001** |
| slip (encoder v ×1.8) | 1.281 | 1.285 | 1.289 | **0.309 ± 0.026** |
| enc_drift | 2.430 | 2.464 | 2.468 | **0.324 ± 0.079** |
| enc_bias | 0.750 | 0.750 | 0.750 | **0.165 ± 0.013** |
| enc_w_bias / enc_w_drift | 0.510 / 1.171 | 0.327 / 0.399 | 0.097 / 0.267 | **0.086 / 0.038** |
| enc_scale / enc_spike / enc_quant | 0.193 / 0.020 / 0.058 | 0.192 / 0.020 / 0.057 | 0.193 / 0.023 / 0.056 | 0.189 / 0.040 / 0.063 |
| enc_stuck | 0.991 | 1.000 | 1.001 | **0.366 ± 0.041** |
| imu_bias / imu_drift | 2.816 / 2.199 | 2.831 / 2.153 | 2.761 / 2.072 | **0.084 / 0.095** |
| imu_dropout | 2.497 | 2.215 | 2.508 | **0.820 ± 0.180** |
| imu_spike | 0.058 | 0.058 | 0.061 | 0.062 |
| combo (lidar sector + slip) | 0.680 | 0.689 | 0.688 | **0.500 ± 0.051** |
| lidar_sector / lidar_noise / lidar_dropout | 0.015 / 0.015 / 0.013 | **0.014 / 0.013 / 0.012** | **0.012 / 0.012 / 0.011** | 0.067 / 0.149 / 0.123 |
| enc_noise / enc_w_noise / imu_noise | 0.062 / 0.036 / 0.107 | 0.062 / 0.027 / 0.110 | 0.061 / 0.012 / 0.108 | 0.073 / 0.028 / **0.045** |
| stationary_imu_bias (정지 중 IMU bias) | 0.454 | 0.456 | 0.504 | **0.107 ± 0.038** |
| stationary_enc_quant (정지 중 encoder quant) | 0.013 | 0.013 | 0.012 | 0.036 |

탐지(fault 채널 평균): AI F1 **0.63** / AUROC **0.86** / sat_frac(R이 R_max 근처에
머문 비율) 11.2% vs Rule F1 0.37 / AUROC 0.73 / sat_frac 0%(규칙 기반은 항상
`s=0.2` 고정값이라 완전 포화되지 않음), RuleZ F1 0.54 / AUROC 0.81. 정상 주행
오탐율(far_clean): AI 2.3% / Rule 2.0% / RuleZ 1.4%. ONNX CPU 추론 15 µs.

AI 후보 비교(전체 데이터, any-fault AUROC): **Autoencoder 0.865**(정상 데이터로만
학습, 재구성 오차를 calibration) > Isolation Forest 0.821. 둘 다 배포되지는
않지만(계획서 11.5절: MLP가 1순위), Autoencoder가 예상보다 잘 나왔다는 점은
정직하게 기록합니다.

### Leave-one-world-out — 한 번도 본 적 없는 월드에서의 일반화

`--holdout <world>`로 그 월드를 통째로 빼고 학습한 뒤 해당 월드에서만 평가
(정상/slip/enc_drift/enc_w_drift/imu_bias/lidar_sector/combo 7개 시나리오 평균 ATE,
`results/holdout_*/summary.md`):

| 학습에서 제외한 월드 | Fixed | Rule | RuleZ | **AI** | far_clean (AI) |
|---|---:|---:|---:|---:|---:|
| tb3_world | 1.239 | 1.151 | 1.087 | **0.263** | 1.9% |
| house | 1.477 | 1.408 | 1.291 | **0.574** | 7.3% |
| stage4 | 0.917 | 0.850 | 0.841 | **0.110** | 6.3% |

한 번도 학습에서 보지 못한 월드에서도 AI가 Fixed 대비 **2.6~8배** ATE를 줄입니다 —
DOF별 신뢰도 추정이 특정 월드의 기하 구조를 암기한 게 아니라 센서 관계의 일반적
패턴을 학습했다는 근거입니다. 오탐율(far_clean)은 main(2.3%)보다 모든 holdout에서
높고, 특히 house/stage4(6~7%)가 tb3_world(1.9%)보다 더 높습니다 — 좁은 통로·
장애물이 많은 환경일수록 처음 보는 기하에서 재보정 여지가 크다는 뜻입니다.

### 정직한 한계

**LiDAR 단독 fault와 순수 noise·spike·quantization형 미세 변화에서는 RuleZ나
Fixed가 AI보다 나은 경우가 많습니다** (`lidar_sector` 0.067 vs Fixed 0.015,
`enc_spike` 0.040 vs Fixed 0.020). 원인을 직접 추적한 결과: Shadow EKF는 encoder·
lidar·imu를 **하나의 공유 상태**로 융합하므로, 라이다 하나만 고장 나도 상태 추정이
오염되고 그 오염된 예측값 기준으로 계산되는 encoder의 innovation/NIS까지 같이
커집니다(실측: `lidar_sector` 구간에서 `s_v_encoder`가 0.56까지 하락 — encoder는
전혀 건드리지 않았는데도). AI는 이 간접 오염 신호와 직접 fault 신호를 구분하지
못해 두 채널을 같이 의심하는 반면, RuleZ는 NIS 크기를 명시적으로 비교해 이 경우를
정확히 처리합니다. 이는 "센서별 Innovation/NIS가 그 센서 고유 상태만 반영한다"는
가정이 융합 필터 구조상 완벽히 성립하지 않는다는 뜻으로, 버그가 아니라 방법론적
한계입니다. **AI의 우위는 체계적(bias/drift/dropout형) fault에서 압도적이고,
귀책이 모호한 단일-센서/순수-noise 상황에서는 아직 명시적 규칙이 유리합니다** —
Fixed vs Rule vs AI 3자 비교라는 프로젝트의 핵심 실험 설계가 의도한 대로, 이
차이를 드러냅니다. (`enc_stuck`은 하이퍼파라미터 튜닝 전에는 Fixed보다 나빴으나,
`RECOVERY_TICKS`/`S_REJECT` 조정 후 2.7배 개선으로 역전되었습니다.)

**하이브리드(채널별로 AI 또는 Rule 선택) 실험 — 검증 후 채택하지 않음.** 위 한계를
보고 "그럼 라이다 채널만 Rule을 쓰면 되지 않나" 싶어 실제로 구현하고
(`asr_core/hybrid.py`) 159개 테스트 에피소드로 검증했습니다. 결과는 **순효과가
마이너스**였습니다 — 채널 자체의 검증셋 성능(F1_gt)만 보면 라이다는 Rule이 확실히
낫지만(omega_lidar F1 0.89 vs AI 0.66), 실제 ATE로 재보면 이야기가 다릅니다:
Rule의 단순 잔차 투표(`RuleBaseline._vote`)도 encoder가 진짜 고장났을 때 그 책임을
라이다 쪽으로 잘못 돌리는 경우가 있어서, 멀쩡한 라이다 측정값을 걷어차 버립니다 —
`enc_drift`가 오히려 2배 이상(0.32→0.68m) 나빠졌습니다. 이는 AI의 encoder 채널이
라이다 고장에 오염되는 것과 정확히 대칭인 문제입니다: **두 방법 다 "이 센서 쌍 중
누가 진짜 범인인가"를 완벽히 가려내지 못하고, 그 실패가 서로 다른 채널에서
나타날 뿐**입니다. 이진(1.0/0.2) 신호를 0.02로 날카롭게 만들어 실제로 거부
threshold를 넘게 하면 dropout 하나(`stage4_08`)는 고쳤지만(1.86→0.32m) `enc_drift`·
`enc_stuck`·`combo`가 더 크게 나빠져 순효과가 오히려 악화됐습니다(23개 시나리오
중 12~16개 개선돼도 나머지 소수의 큰 악화가 전부 상쇄). **따라서 배포 기본값은
순수 AI(`mode:=ai`)로 유지합니다** — `asr_core/hybrid.py`와 `mode:=hybrid`는
코드·테스트·`asr_evaluation.compare`의 `ate_hybrid` 컬럼으로 남겨뒀지만 기본값은
아닙니다. "채널별로 무조건 더 단순한 쪽을 쓴다"는 직관적으로 맞아 보이는 아이디어가
실제로는 net-negative였다는 것 자체가, 이 프로젝트가 안고 있는 근본 문제(상관된
센서 쌍의 귀책 모호성)가 방법론을 바꾼다고 쉽게 없어지지 않는다는 걸 보여주는
정량적 증거입니다.

**개발 중 네 개의 실버그를 발견해 고쳤습니다** (전부 아래에서 재사용):

1. `StationaryDetector`가 완전히 무력화돼 있었습니다 — 가속도 임계값(0.30 m/s²)이
   Gazebo 시뮬레이션 IMU의 실제 노이즈(순간값 std 1.1~3.8 m/s²)보다 훨씬 타이트해서,
   확실히 8초간 정지한 구간에서도 감지율이 0%였습니다. `||a||`에 짧은 EMA 스무딩을
   추가하고 임계값을 1.5로 재보정했으며, 회귀 테스트
   (`test_stationary_detector_survives_imu_noise`)를 추가했습니다. 이 버그가 있던
   상태에서는 정지 중 fault를 과소평가하지 않는다는 계획서 10절의 설계 의도를 실제로
   검증할 수 없었습니다.
2. 라이브 검증 중 발견: `asr_demo.launch.py`의 스폰 위치가 장애물에 가까워, 스크립트
   드라이버(`driver_node.py`)의 회피 로직(장애물 감지 시 제자리 회전만 함)이 오목한
   구석에서 영구적으로 데드락에 빠질 수 있었습니다 — 실측: 80초 내내 0.4m×0.4m 박스
   안에서만 회전(순 이동 거의 0). 오프라인 학습에 쓴 32개 run은 각기 다른 seed로 전부
   2.3~4.5m를 정상 탐색해 이 버그의 영향을 받지 않았지만(데이터셋 결과는 유효), 데모
   스폰 위치와 결합된 특정 seed는 걸릴 수 있음을 라이브 실행에서 처음 발견했습니다.
   회피 3초 초과 시 후진하도록 고쳤고, `asr_sensing/test/test_driver.py` 회귀
   테스트를 추가했습니다.
3. `smoke_test_live`(아래)로 10개 시나리오를 라이브로 돌려보다가 발견: 온라인 데모용
   fault injector(`fault_injector_node.py`)가 오프라인 코드(`asr_core.faults`)와
   다르게 구현돼 있어서, `enc_w_bias`/`enc_w_drift`(encoder omega 채널 fault)가
   실제로는 **아무 효과가 없었습니다** — 양쪽 바퀴에 같은 bias를 더해서 차이(omega)는
   그대로 두고 평균(v)만 바뀌는 실수였습니다. 오른쪽 바퀴는 +, 왼쪽 바퀴는 -로 반대
   부호를 주도록 고쳐 차이(omega)만 움직이고 평균(v)은 그대로 두게 했습니다
   (`wheel_fault_velocity`, 순수함수로 분리해 `test_omega_fault_shifts_difference_not_average`
   회귀 테스트 추가).
4. 같은 방식으로 발견: `imu_dropout` 시나리오가 온라인에서 **아예 구현이 안 돼
   있었습니다**(bias/noise만 있고 dropout 분기가 없어서 메시지가 그대로 통과). IMU가
   먹통일 때의 마지막 값을 얼려서 계속 발행하도록 추가했습니다.
   이 두 버그(3, 4)는 오프라인 데이터셋 결과에는 영향이 없습니다 — offline
   `asr_core.faults`는 처음부터 올바르게 구현돼 있었고, 문제는 온라인 데모 전용
   코드에만 있었습니다.

**정지 상태 전용 실험**(계획서 19절 실험 7): 각 run의 ground truth에서 실제로
멈춘 구간(중앙값 4.3초)을 찾아 그 구간에만 fault를 주입합니다. `stationary_imu_bias`는
AI가 Fixed 대비 4.2배 개선 — 위 버그 1이 있던 상태에서는 검증 불가능했던 결과입니다.
반대로 `stationary_enc_quant`는 두 방법 모두 절대 오차가 이미 작아(0.01~0.04m)
개선 폭이 뚜렷하지 않습니다.

**라이브 데모 검증** (버그 2를 고친 뒤, Gazebo 실시간·헤드리스
`asr_demo.launch.py scenario:=slip`, 99초): reliability 20.0 Hz 유지, slip 중
`s_v_encoder` 0.87→0.20으로 정확히 하락하면서 `s_v_lidar` 0.97·`s_ω_imu` 0.86은
거의 그대로 유지됩니다 — DOF 분리 설계가 라이브 환경에서도 의도대로 동작함을
확인했습니다. 다만 이 한 에피소드의 ATE는 fixed 1.65 vs adaptive 1.74로 뚜렷한
우위가 없었습니다(`asr_evaluation.analyze_live`로 재현) — 730개 에피소드를 평균한
데이터셋 결과와 달리, 장애물이 많은 단일 라이브 주행 하나는 노이즈가 크고 회피
기동 자체의 난이도가 지배적이라는 뜻으로, 신뢰도 있는 성능 주장은 데이터셋 기반
결과(위 표)를 근거로 삼아야 합니다. 대시보드 GUI는 Xvfb + 실제 노드 실행으로
화면을 직접 캡처해 확인했습니다(reliability 곡선·GT/Fixed/Adaptive 궤적·상태
텍스트 정상 렌더링).

버그 3, 4를 고친 뒤 `smoke_test_live --all`(10개 시나리오: slip·enc_drift·enc_bias·
enc_w_bias·enc_w_drift·imu_bias·imu_drift·imu_dropout·lidar_sector·lidar_dropout)이
**전부 PASS**합니다 — 각 시나리오의 fault 채널이 실제로 하락하는지를 라이브로
자동 확인하는 유일한 종단 테스트입니다. `imu_cb`의 dropout-freeze 로직은 이후
`wheel_fault_velocity`와 같은 방식으로 순수함수(`imu_fault_step`)로 분리해 빠른
단위테스트 2개를 추가했습니다(`asr_sensing/test/test_driver.py`). `imu_dropout`은
45초 녹화 구간과 gzserver 타이밍이 겹치는 정도가 반복 실행마다 흔들려 `MIN_GAP`을
0.10으로 보정했습니다(lidar_dropout과 달리 "원래 약한 신호"가 아니라 경계선
노이즈 — 재현 실행으로 리팩터 회귀가 아님을 먼저 확인한 뒤 조정). 같은 흐름으로
`asr_reliability_estimator`에 있던 테스트 공백도 메웠습니다: 온라인 시각동기화
링버퍼(`Series`, plan 18절)는 지금까지 아무 테스트도 없었는데, ROS에 의존하지
않는 순수 로직이라 빠른 단위테스트 7개로 분리했습니다(`asr_reliability_estimator
/test/test_feature_builder.py`). `covariance_injector._f`의 covariance-inflation
수식도 처음엔 별도 함수로 뽑아 테스트했지만, 곧 `asr_core.smoothing.inflate`와
정확히 같은 공식을 중복 구현한 것뿐임을 깨닫고(ponytail-audit) `_f`가 `inflate`를
그대로 호출하도록 되돌리고 중복 테스트 파일은 지웠습니다 — 새로 발견한 케이스
(범위 밖 `s`, 예: IMU의 미사용 v-채널 센티널 -1.0의 클리핑)만 `asr_core/test
/test_core.py`의 기존 `test_inflation_bounded`에 합쳤습니다. 전체 유닛테스트는
이제 3개 패키지에 걸쳐 총 25개(`asr_core` 12 + `asr_sensing` 6 +
`asr_reliability_estimator` 7) 전부 통과합니다.

외부 데이터 검사(`asr_evaluation.external_check`, `~/ros2_auto_tuner` TB3 백):
IMU/GT가 없어 학습·ATE 평가엔 못 쓰고 encoder–LiDAR 잔차만 비교했으며, 실린더
필드 세계에서는 ICP v가 과소추정돼 잔차 std가 sim 대비 여러 배 — 다른 환경으로
옮길 때 `NOMINAL_STD` 재캘리브레이션이 필요함을 뜻합니다.

그래프: `results/ate_by_scenario.png`, `results/main/episode_slip.png`,
`results/main/episode_combo.png`.

## Docker

```bash
docker build -f docker/Dockerfile.runtime -t asr:runtime .   # 경량, 실로봇 배포용
docker build -f docker/Dockerfile.dev -t asr:dev .           # Gazebo+PyTorch 재현/재학습용
```

`Dockerfile.runtime`은 Humble·**Jazzy 둘 다 빌드해서 실제로 컨테이너를 실행해**
확인했습니다(`--build-arg ROS_DISTRO=jazzy`). Jazzy(Ubuntu 24.04)는 PEP 668
(externally-managed-environment)를 강제해서 원래 `pip3 install`이 그대로 실패했는데
— Humble(22.04)의 구버전 pip는 `--break-system-packages` 플래그 자체를 모르기 때문에
무조건 그 플래그를 넣을 수도 없어서, 플래그 없이 먼저 시도하고 실패하면
`--break-system-packages`로 재시도하는 방식으로 두 배포판 모두에서 빌드되게
고쳤습니다. 계획서 31.5절이 명시한 "배포용 패키지는 Jazzy를 기본 타겟으로 한다"를
실제로 검증한 결과입니다.

## 라이선스

Apache-2.0
