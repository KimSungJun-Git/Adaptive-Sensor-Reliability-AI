# Adaptive Sensor Reliability AI 개발계획서

## 1. 프로젝트 개요

### 1.1 프로젝트명

**Adaptive Sensor Reliability AI**  
부제: **다중 센서 일관성 기반 실시간 신뢰도 추정 및 적응형 EKF**

### 1.2 한 줄 설명

LiDAR Odometry, IMU, Wheel Encoder의 측정 관계와 독립적인 Shadow EKF의 Innovation/NIS를 AI가 분석하여 센서별·자유도별 신뢰도를 추정하고, 그 결과를 EKF의 Measurement Covariance에 동적으로 반영하여 센서 이상 상황에서도 안정적인 상태 추정을 수행한다.

### 1.3 프로젝트 목적

자율주행 로봇은 여러 센서를 동시에 사용하지만 센서의 정확도는 주행 상황에 따라 달라진다. 대표적으로 Wheel Slip이 발생하면 Wheel Encoder의 선속도 추정은 크게 틀릴 수 있지만, 동일 센서에서 얻는 회전 정보는 여전히 유효할 수 있다.

기존 EKF는 여러 센서를 융합할 수 있지만, 실제 센서 상태가 시간에 따라 변하는 상황에서는 센서 공분산을 고정적으로 설정하는 방식만으로 모든 상황에 최적으로 대응하기 어렵다.

본 프로젝트는 이 문제를 다음과 같이 해결한다.

> **EKF는 상태 추정을 담당하고, AI는 현재 각 센서 관측을 얼마나 신뢰할 수 있는지를 추정한다.**

AI가 센서 원본 데이터를 임의로 수정하는 대신, AI가 추정한 Reliability를 EKF의 측정 공분산에 반영하여 신뢰도가 떨어진 측정값의 영향력을 줄인다.

---

## 2. 해결하려는 문제

### 2.1 기존 방식의 문제

일반적인 센서 융합 구조는 다음과 같다.

```text
LiDAR ─┐
IMU ───┼──> EKF ──> Fused State
Encoder┘
```

센서별 Measurement Covariance `R`이 적절하게 설정되어 있더라도 실제 센서 특성은 고정되어 있지 않다.

예:

```text
정상 주행
Encoder 정확도 높음

        ↓

Wheel Slip 발생
Encoder 선속도 오차 증가
```

그러나 EKF가 동일한 Encoder covariance를 계속 사용하면 잘못된 측정값이 상태 추정에 지속적으로 영향을 줄 수 있다.

### 2.2 핵심 문제 정의

본 프로젝트가 해결하는 문제는 다음과 같다.

> **센서 자체의 고장 여부를 단순히 0/1로 판단하는 것이 아니라, 현재 상황에서 각 측정값을 얼마나 신뢰해야 하는지를 지속적으로 추정하고 이를 센서 융합 과정에 반영한다.**

---

## 3. 핵심 아이디어

### 3.1 핵심 개념

**Sensor Reliability Estimation + Adaptive Covariance Scaling**

AI의 역할:

```text
"이 센서가 지금 얼마나 믿을 만한가?"
```

EKF의 역할:

```text
"그 신뢰도를 반영해 현재 상태를 어떻게 추정할 것인가?"
```

### 3.2 기존 방식과의 차이

| 구분 | 기존 EKF | 본 프로젝트 |
|---|---|---|
| 상태 추정 | EKF | EKF |
| 센서 융합 | O | O |
| 센서 이상 판단 | 제한적 통계 게이팅 | AI 기반 신뢰도 추정 |
| 센서 신뢰도 | 정적 covariance 중심 | 시간에 따라 동적 변화 |
| 신뢰도 단위 | 센서/측정 단위 | 센서×DOF 조합 단위 (`s_v_encoder`, `s_ω_imu` 등) |
| covariance | 고정 설정 | AI 기반 동적 조정 |
| AI 역할 | 없음 | 신뢰도 추정 |
| Nav2 | 별도 시스템 | 사용하지 않음 |

---

## 4. 시스템 범위

### 4.1 포함 범위

- Wheel Encoder
- IMU
- 2D LiDAR
- LiDAR Scan Matching 기반 LiDAR Odometry
- 센서 시간 동기화
- 센서 간 Residual 계산
- Fixed Shadow EKF
- Innovation 및 NIS 계산
- 시계열 Feature 생성
- AI 기반 Reliability 추정
- `s_v`, `s_ω` 자유도 분리
- Bounded Covariance Inflation
- Adaptive EKF
- Fault Injection
- 정량 평가
- 실시간 시각화

### 4.2 제외 범위

- Nav2
- Path Planning
- Obstacle Avoidance
- SLAM 자체 개발
- Camera / Vision
- LLM
- 챗봇
- 복잡한 웹 서비스

Nav2는 본 프로젝트의 핵심 목적과 직접적인 관계가 없으므로 범위에서 제외한다. 본 프로젝트는 센서 신뢰도와 상태 추정까지를 검증하며, 최종 출력은 Fused Odometry/State이다.

---

## 5. 센서 구성

### 5.1 Wheel Encoder

Wheel Encoder에서 다음 값을 계산한다.

- 선속도 `v_encoder`
- 각속도 `ω_encoder`

차동 구동 기준으로:

```text
v = (v_right + v_left) / 2
ω = (v_right - v_left) / wheel_base
```

### 5.2 IMU

IMU는 다음 정보를 사용한다.

- 선가속도 `a_x`
- 각속도 `ω_z`
- 필요 시 자세/중력 보정 정보

주의사항:

IMU 가속도를 장시간 적분하여 독립적인 절대 속도 기준으로 사용하는 것은 Drift 때문에 부적절하다. 따라서 IMU는 주로 단기 운동 변화, 회전 정보, EKF Innovation/NIS 및 시계열 Feature에 활용한다.

### 5.3 LiDAR

LiDAR 원시 Scan 자체를 최종 EKF measurement로 바로 사용하는 대신 연속 Scan Matching을 통해 상대적인 이동량을 추정한다.

```text
Scan(t-1)
    +
Scan(t)
    ↓
Scan Matching
    ↓
Δx, Δy, Δyaw
    ↓
LiDAR Odometry
    ↓
v_lidar, ω_lidar 또는 Δpose
```

Scan Matching은 직접 구현하는 대신, 이미 구축된 SLAM 스택(예: Cartographer)의 local scan matching 결과를 LiDAR Odometry로 그대로 활용하는 방법을 우선 검토한다 — 핵심 검증 대상인 Reliability AI/Adaptive EKF 쪽에 구현 자원을 집중할 수 있다. 직접 구현이 필요한 경우 2D 전용 방식(PL-ICP/CSM, `laser_scan_matcher` 등)을 우선 검토한다. KISS-ICP는 원래 3D LiDAR 포인트클라우드를 대상으로 설계된 방식이라 2D Scan에 적용하려면 추가적인 adaptation이 필요하다.

---

## 6. 전체 시스템 아키텍처

```text
                    ┌────────────┐
                    │   LiDAR    │
                    └─────┬──────┘
                          │
                    LiDAR Odometry
                          │
┌──────────┐             │
│ Encoder  │─────────────┤
└────┬─────┘             │
     │                    │
     v, ω                 │
┌────▼─────┐             │
│   IMU    │─────────────┤
└──────────┘             │
                         ▼
               ┌──────────────────┐
               │ Time Synchronizer│
               └────────┬─────────┘
                        ↓
               ┌──────────────────┐
               │ Feature Builder  │
               │                  │
               │ Residual         │
               │ Innovation       │
               │ NIS              │
               │ Temporal Stats   │
               │ Stationary Flag  │
               └────────┬─────────┘
                        ↓
               ┌──────────────────┐
               │ Reliability AI   │
               │                  │
               │ s_v_encoder      │
               │ s_ω_encoder      │
               │ s_v_lidar        │
               │ s_ω_lidar        │
               │ s_ω_imu          │
               └────────┬─────────┘
                        ↓
             Asymmetric Smoothing
                        ↓
             Bounded R Inflation
                        ↓
               ┌─────────────────┐
               │   Adaptive EKF  │
               └────────┬────────┘
                        ↓
                    Fused State
                        ↓
               ATE / RPE / RMSE
```

---

## 7. 핵심 설계 ① — DOF 분리형 Reliability

센서 전체를 하나의 스칼라 Reliability로 평가하지 않는다. 또한 동일 DOF라도 센서가 다르면 서로 다른 Reliability를 가질 수 있어야 하므로, Reliability는 **DOF 단위가 아니라 (센서 × DOF) 조합 단위**로 정의한다.

### 7.1 기본 출력

```text
s_v_encoder ∈ [0, 1]      s_ω_encoder ∈ [0, 1]
s_v_lidar   ∈ [0, 1]      s_ω_lidar   ∈ [0, 1]
                          s_ω_imu     ∈ [0, 1]
```

- 각 센서가 실제로 제공하는 DOF에 대해서만 Reliability를 산출한다 (IMU는 `v`를 직접 제공하지 않으므로 `s_v_imu`는 정의하지 않음)
- 이하 문서에서 편의상 "`s_v`, `s_ω`"로 표기한 부분은 위 (센서×DOF) 벡터 전체를 통칭하는 축약 표기로 이해한다.

### 7.2 이유

Wheel Slip 상황에서 Encoder의 선속도는 크게 왜곡되어도, 좌우 바퀴 속도 차이로 계산되는 Encoder의 각속도는 비교적 유효할 수 있고, 동시에 LiDAR Odometry가 제공하는 선속도는 전혀 영향받지 않는다.

따라서:

```text
Wheel Slip
    ↓
 s_v_encoder ↓↓↓
 s_ω_encoder ───
 s_v_lidar   ───   (영향 없음)
```

와 같이 센서별로 분리되어야 한다. 만약 `s_v`가 센서 구분 없이 전역 값 하나였다면, Encoder v의 저하가 정상인 LiDAR v의 covariance까지 함께 부풀려 "멀쩡한 센서를 억울하게 의심"하는, 본 프로젝트가 원래 피하려던 문제를 그대로 재현하게 된다.

이 분리를 통해 정상적인 회전 정보와 다른 센서의 정상적인 정보를 불필요하게 폐기하지 않는다.

---

## 8. 핵심 설계 ② — Shadow EKF

### 8.1 필요성

Adaptive EKF의 출력으로부터 Innovation을 계산하여 다시 AI 입력으로 사용하면 다음과 같은 피드백 루프가 발생할 수 있다.

```text
AI Reliability
      ↓
Adaptive EKF R 변경
      ↓
EKF Prediction 변경
      ↓
Innovation 변경
      ↓
AI Reliability
```

이러한 구조에서는 AI가 일시적인 이상을 영구적인 고장으로 판단하는 Reliability Latching이 발생할 수 있다.

### 8.2 해결 방법

AI 입력용 Innovation/NIS는 **Adaptive EKF가 아닌 고정 공분산의 Shadow EKF**에서 계산한다.

```text
                    ┌───────────────┐
Sensors ───────────>│ Shadow EKF    │
                    │ Fixed R       │
                    └──────┬────────┘
                           ↓
                    Innovation / NIS
                           ↓
                     Reliability AI
                           ↓
                     Adaptive EKF
```

### 8.3 역할 분리

**Shadow EKF**

- 항상 동일한 nominal covariance 사용
- AI 입력용 기준 상태 생성
- Adaptive feedback 차단

**Adaptive EKF**

- AI Reliability를 이용해 measurement covariance를 조절
- 최종 상태 추정 담당

### 8.4 주의 — LiDAR Odometry 경유 우회 피드백

Innovation 경로는 Shadow EKF로 차단했지만, LiDAR Scan Matching의 초기 정합 guess(initial guess)가 Adaptive EKF의 이전 pose 추정치를 사용하면 그 경로로 간접적인 피드백이 다시 발생할 수 있다.

```text
Adaptive EKF pose
      ↓ (초기 guess로 사용 시)
Scan Matching
      ↓
LiDAR Odometry
      ↓
Shadow EKF 입력
```

따라서 Scan Matching의 초기 guess는 Adaptive EKF가 아니라 Shadow EKF의 예측 상태 또는 고정 모션 모델(constant velocity 등)에서 가져온다. 이 경로를 명시적으로 분리해야 Shadow EKF의 독립성이 실제로 보장된다.

---

## 9. 핵심 설계 ③ — Feature Engineering

AI는 센서 원본만 직접 입력받지 않고 센서 관계를 나타내는 Feature를 사용한다.

### 9.1 센서 간 Residual

예:

\[
r_{EL}=v_{encoder}-v_{lidar}
\]

\[
r_{EI}=v_{encoder}-v_{imu,short}
\]

\[
r_{LI}=v_{lidar}-v_{imu,short}
\]

각속도:

\[
r_{\omega,EI}=\omega_{encoder}-\omega_{imu}
\]

주의: IMU의 속도는 장기 적분값을 사용하지 않고 단기 동역학 정보 또는 EKF 기반 값으로 구성한다.

### 9.2 Shadow EKF Innovation

\[
\nu_k=z_k-H_k\hat{x}_k^-
\]

- `z_k`: 현재 센서 측정
- `H_k`: Measurement Model
- `x^-_k`: Shadow EKF의 예측 상태

### 9.3 NIS

\[
NIS=\nu_k^TS_k^{-1}\nu_k
\]

NIS는 현재 측정값이 Shadow EKF가 예상하는 상태와 비교하여 얼마나 비정상적인지 나타내는 통계량으로 활용한다.

### 9.4 Temporal Features

최근 일정 시간 Window에 대해 다음 Feature를 계산한다.

- residual mean
- residual standard deviation
- residual absolute mean
- residual slope
- innovation mean
- NIS mean
- NIS maximum
- NIS variance
- 최근 Reliability 변화량
- stationary 여부

---

## 10. 핵심 설계 ④ — Stationary / Zero-Velocity 처리

정지 상태에서는 IMU noise, encoder quantization, 구조물 미세 진동이 상대적으로 크게 보일 수 있으므로 일반 주행과 동일한 기준으로 Reliability를 계산하지 않는다.

### 10.1 Stationary 조건 예시

```text
|v_encoder| < threshold_v
AND
|ω_imu| < threshold_ω
AND
||a|| - g| < threshold_a
```

조건이 일정 시간 이상 유지되면:

```text
STATIONARY = TRUE
```

### 10.2 처리

Stationary 상태에서는:

- IMU의 작은 noise를 fault로 과대평가하지 않음
- 저속 encoder quantization을 완화
- Reliability 변화 폭을 제한
- 필요 시 Zero-Velocity Update(ZUPT) 적용

---

## 11. 핵심 설계 ⑤ — AI 모델

### 11.1 1차 기준 모델

**Rule-based Threshold**

```text
|residual| > threshold
→ anomaly
```

목적:

AI가 실제로 기존 단순 방법보다 개선되는지 비교하기 위한 baseline이다.

### 11.2 AI 후보 1 — Isolation Forest

장점:

- 구현이 간단함
- 정상 데이터 중심 학습 가능
- CPU에서도 빠름
- baseline보다 복잡한 패턴 탐지 가능

단점:

- 시계열 자체를 직접 모델링하는 능력이 제한적

### 11.3 AI 후보 2 — MLP Regression

Ground Truth 기반으로 생성한 Reliability Label을 목표값으로 학습한다.

```text
Input Features
    ↓
Dense 64
    ↓
ReLU
    ↓
Dense 32
    ↓
ReLU
    ↓
Dense 16
    ↓
ReLU
    ↓
Output
[s_v_encoder, s_ω_encoder,
 s_v_lidar, s_ω_lidar, s_ω_imu]
```

### 11.4 AI 후보 3 — Autoencoder

정상 데이터의 시계열 Feature를 학습하여 Reconstruction Error를 계산한다.

```text
Normal Data
    ↓
Encoder
    ↓
Latent
    ↓
Decoder
    ↓
Reconstruction Error
    ↓
Anomaly Score
```

다만 Autoencoder의 Reconstruction Error 자체는 물리적인 단위가 없으므로, 이를 그대로 공분산에 사용하는 것이 아니라 검증 데이터에서 calibration 과정을 거친다.

### 11.5 최종 모델 선정 원칙

초기에는 **MLP + temporal feature**를 1순위 후보로 검토하고, Isolation Forest와 Autoencoder를 비교군으로 둔다. 데이터 규모가 크지 않을 경우 **Gradient Boosting(LightGBM 등)**을 MLP의 대안 후보로 추가 검토한다 — 정형 시계열 feature(residual/NIS 통계량)에는 소규모 데이터에서도 MLP보다 안정적인 경우가 많고, feature importance로 해석도 쉽다.

모델의 복잡도보다 다음을 우선한다.

1. Fault Detection 성능
2. False Alarm Rate
3. Detection Latency
4. Reliability 안정성
5. Inference 비용

---

## 12. 학습 Label 생성

### 12.1 Ground Truth의 사용 목적

Gazebo의 Ground Truth는 **실시간 추론용이 아니라 학습 Label 및 성능 평가용**으로 사용한다.

### 12.2 Sensor Error

각 센서 `i` (encoder, lidar, imu 등)가 제공하는 DOF에 대해 Ground Truth와 비교한다.

\[
e_{v,i}=|v_{i}-v_{GT}|
\]

\[
e_{\omega,i}=|\omega_{i}-\omega_{GT}|
\]

### 12.3 Reliability Label

공칭 센서 표준편차를 이용해 물리적인 의미를 갖는 Reliability Label을 센서별로 생성한다.

\[
s_{v,i}=\exp\left(-\frac{e_{v,i}^2}{2\sigma_{v,i}^2}\right)
\]

\[
s_{\omega,i}=\exp\left(-\frac{e_{\omega,i}^2}{2\sigma_{\omega,i}^2}\right)
\]

여기서 `σ_i`는 센서 `i`별로 실험을 통해 실제 정상 오차 분포에서 추정한다. 예를 들어 `s_v_encoder`는 `σ_v_encoder`를, `s_v_lidar`는 `σ_v_lidar`를 사용한다.

단순히 임의의 0~1 정규화를 적용하지 않는다.

### 12.4 주의점

Ground Truth 기반 Label이 실제 환경에서도 절대적인 정답이라고 가정하지 않는다. 따라서 Sim-to-Real 검증에서 다양한 disturbance를 사용하고, 실제 데이터에서 분포 이동 여부를 확인한다.

---

## 13. Fault Injection 설계

실제 센서 고장 데이터를 충분히 확보하기 어려우므로 정상 Gazebo 데이터를 기반으로 현실적인 고장 패턴을 주입한다.

### 13.1 Encoder Fault

- Bias
- Scale error
- Drift
- Spike
- Stuck
- Gaussian noise increase
- Wheel Slip
- Quantization 증가

### 13.2 IMU Fault

- Bias
- Bias drift
- Noise increase
- Spike
- Dropout

### 13.3 LiDAR Fault / Degradation

- Scan dropout
- Range noise
- Partial scan corruption
- Feature-poor environment
- Scan matching quality degradation

### 13.4 중요한 원칙

극단적인 오류값을 넣어 쉽게 탐지되도록 만들지 않는다.

나쁜 예:

```text
Encoder = 9999
```

좋은 예:

```text
정상: 1.00, 1.02, 1.01, 1.03
Drift: 1.00, 1.03, 1.06, 1.10, 1.14
```

목표는 **사람이 단순 threshold로 잡기 어려운 미세한 이상**을 평가하는 것이다.

---

## 14. Reliability Smoothing

AI 출력 `s_raw`를 EKF에 즉시 연결하지 않는다.

### 14.1 비대칭 EMA

\[
s_t=
\begin{cases}
\gamma_{fall}s_{t-1}+(1-\gamma_{fall})s_{raw,t}, & s_{raw,t}<s_{t-1}\\
\gamma_{rise}s_{t-1}+(1-\gamma_{rise})s_{raw,t}, & s_{raw,t}\ge s_{t-1}
\end{cases}
\]

### 14.2 목적

- Fault 발생: 빠르게 Reliability 하락
- Fault 종료: 정상 여부를 확인하며 안정적으로 회복
- 순간적인 센서 spike에 의한 채터링 방지

`γ_fall`, `γ_rise`는 초기값을 임의로 최종 확정하지 않고 실험으로 조정한다.

### 14.3 개선된 Recovery Gate

시간만 지나면 복구시키는 방식 대신 다음 조건을 함께 사용한다.

```text
Residual 정상
AND
NIS 정상
AND
Sensor consistency 정상
```

이 조건이 일정 구간 지속될 경우에만 Recovery를 허용한다.

---

## 15. Adaptive Covariance Scaling

AI가 출력한 Reliability를 그대로 `1/s`에 대입하지 않는다.

초기 구현은 다음과 같이 단순하고 bounded한 형태를 사용한다.

\[
R_{adapted}=R_0[1+\alpha(1-s)]
\]

그리고 반드시:

\[
R_{min}\le R_{adapted}\le R_{max}
\]

를 적용한다.

### 15.1 의미

```text
s ≈ 1
→ 기존 covariance 유지

s 감소
→ covariance 증가
→ 센서 영향 감소

s가 매우 낮음
→ 최대 inflation에서 제한
```

### 15.2 센서×DOF별 적용

```text
s_v_encoder 낮음
→ R_v_encoder 증가        (Encoder 선속도 측정만 영향 감소)

s_ω_encoder 정상
→ R_ω_encoder 유지

s_v_lidar 정상
→ R_v_lidar 유지          (LiDAR 선속도는 그대로 신뢰)
```

각 센서의 covariance는 그 센서 자신의 reliability로만 조정되며 다른 센서의 covariance에 영향을 주지 않는다. 따라서 Wheel Slip 발생 시 Encoder 선속도 정보의 영향만 줄이고, LiDAR 선속도와 두 센서의 회전 정보는 그대로 유지할 수 있다.

### 15.3 Measurement Rejection

Reliability가 매우 낮아도 무조건 측정을 제거하지 않는다.

예시 정책:

```text
s > 0.7
→ 정상 반영

0.3 < s ≤ 0.7
→ covariance 증가

0.1 < s ≤ 0.3
→ 강한 covariance 증가

s ≤ 0.1
→ rejection 후보
```

실제 threshold는 실험을 통해 결정한다.

---

## 16. EKF 구성

### 16.1 역할

Adaptive EKF는 최종 상태 추정을 담당한다.

예시 상태벡터:

\[
x=[x,y,\theta,v,\omega]^T
\]

실제 상태 구성은 사용 센서 및 구현 방식에 따라 조정한다.

### 16.2 두 개의 필터

#### Shadow EKF

```text
고정 R
↓
Innovation / NIS 생성
```

#### Adaptive EKF

```text
AI Reliability
↓
Dynamic R
↓
최종 상태 추정
```

### 16.3 안전 원칙

AI가 센서값 자체를 임의로 수정하지 않는다.

```text
센서값
 ↓
AI
 ↓
"얼마나 믿을 것인가"
 ↓
Covariance 조정
 ↓
EKF
```

최종 상태 추정은 기존의 검증된 EKF 구조에서 수행한다.

---

## 17. 데이터 파이프라인

```text
Gazebo
  ↓
ROS2 sensor topics
  ↓
rosbag2 recording
  ↓
Time Synchronization
  ↓
LiDAR Odometry 생성
  ↓
Ground Truth Alignment
  ↓
Feature Extraction
  ↓
Fault Injection
  ↓
Dataset
  ├── train
  ├── validation
  └── test
```

### 권장 데이터 구조

```text
data/
├── raw/
├── synchronized/
├── features/
├── faults/
├── train/
├── validation/
└── test/
```

### Train/Val/Test 분할 원칙

Fault를 시간에 따라 점진적으로 주입하므로(drift 등) 인접 timestep은 서로 강하게 상관되어 있다. 샘플(timestep) 단위로 무작위 분할하면 거의 동일한 시계열 윈도우가 train/test 양쪽에 섞여 성능이 과대평가된다. 따라서 분할은 **run(주행 에피소드) 단위**로 수행하고, 동일 run이 train/validation/test에 걸쳐 나뉘지 않도록 한다.

---

## 18. 시간 동기화

센서 주기가 다르므로 단순한 동일 timestamp 매칭을 사용하지 않는다.

예시:

```text
IMU       100 Hz
Encoder    50 Hz
LiDAR      10 Hz
```

### 처리 원칙

- 과거 Ring Buffer 유지
- 각 센서 timestamp 기준 상태 조회
- LiDAR scan matching 시점에 IMU/Encoder 값을 interpolation
- 각 데이터의 실제 timestamp를 보존
- 측정 도착 시간과 센서 측정 시간이 다른 경우 latency 기록

### 목표

가속/감속 구간에서 시간 차이 때문에 발생하는 허위 residual을 최소화한다.

---

## 19. 실험 시나리오

### 실험 1 — 정상 주행

목적:

- False Alarm 측정
- Reliability 안정성 확인

### 실험 2 — Encoder Wheel Slip

```text
정상
→ Wheel Slip
→ 정상 복귀
```

관찰:

- `s_v` 하락
- `s_ω` 유지 여부
- Encoder covariance 증가
- Adaptive EKF 오차 변화

### 실험 3 — Encoder Drift

완만한 오류 증가를 주입한다.

목적:

- 단순 threshold 대비 AI의 조기 탐지 성능 확인

### 실험 4 — IMU Bias

목적:

- IMU 이상 상황에서 다른 센서와의 관계 분석

### 실험 5 — LiDAR Degradation

목적:

- Feature-poor 상황
- Scan Matching 품질 하락
- LiDAR reliability 변화 확인

### 실험 6 — 복합 고장

예:

```text
LiDAR degradation
+
Encoder slip
```

목적:

단순 다수결 방식이 정상 IMU를 오판할 가능성을 평가한다.

### 실험 7 — Stationary

목적:

- 정지 상태에서 false alarm 발생 여부 확인
- IMU bias/noise 및 encoder quantization 영향 검증

---

## 20. 비교군

최종 실험은 최소 3가지 방법을 비교한다.

### A. Fixed EKF

```text
고정 covariance
→ EKF
```

### B. Rule-based Adaptive EKF

```text
Residual threshold
→ covariance 조정
→ EKF
```

### C. AI-Adaptive EKF

```text
AI Reliability
→ covariance 조정
→ EKF
```

핵심 질문은 다음과 같다.

> **AI 기반 Reliability가 단순 규칙 기반 적응보다 실제 상태 추정 성능을 개선하는가?**

---

## 21. 평가 지표

### 21.1 상태 추정 성능

- ATE (Absolute Trajectory Error)
- RPE (Relative Pose Error)
- RMSE

### 21.2 Fault Detection

- Precision
- Recall
- F1-score
- AUROC

### 21.3 시간 성능

- Detection Latency
- Recovery Time

### 21.4 안정성

- False Alarm Rate
- Reliability Chattering
- Reliability variance
- Covariance saturation 빈도

### 21.5 최종 핵심 결과

가장 중요한 비교는 다음과 같다.

```text
Fixed EKF
vs
Rule-based Adaptive EKF
vs
AI-Adaptive EKF
```

그리고 **Fault 상황에서 최종 ATE/RPE가 얼마나 감소하는지**를 핵심 결과로 제시한다.

---

## 22. 성공 기준

프로젝트의 성공 여부를 단순히 AI Accuracy로 판단하지 않는다.

최소한 다음을 만족하는 것을 목표로 한다.

### AI 수준

- fault pattern을 정상 noise와 구분
- 센서 reliability가 시간에 따라 안정적으로 변화
- 정상 복귀 시 과도한 지연이나 채터링이 없음

### EKF 수준

- 센서 fault 발생 시 최종 상태 추정 오차 증가 억제
- 정상적인 DOF 정보가 불필요하게 제거되지 않음
- covariance가 bounded 상태를 유지

### 시스템 수준

- Fixed EKF 대비 fault 상황 성능 개선
- Rule-based Adaptive EKF 대비 AI 방식의 추가 이점 확인
- 실시간 처리 가능

---

## 23. 개발 단계

아래 Phase는 순서대로 진행하되, 시간 제약(해커톤 데모 등)이 있는 경우 다음을 최소 경로로 삼는다: Phase 1~3, 5~7(Rule-based + MLP까지만), 8~9, 10(Wheel Slip 시나리오만), 11. Autoencoder 비교, IMU/LiDAR degradation, 복합 fault 시나리오는 28절의 "선택 구현"에 해당하며 시간이 남을 때 확장한다.

### Phase 1 — 기본 데이터 확보

1. Gazebo 환경 구성
2. LiDAR / IMU / Encoder / Ground Truth 기록
3. rosbag2 저장
4. 각 센서 timestamp 확인

**완료 조건**

- 정상 주행 데이터셋 확보
- Ground Truth와 모든 센서 데이터 정렬 가능

---

### Phase 2 — 센서 전처리

1. Encoder에서 `v`, `ω` 계산
2. IMU 전처리
3. LiDAR Scan Matching 구현
4. LiDAR Odometry 생성
5. 시간 동기화 구현

**완료 조건**

```text
LiDAR Odometry
IMU
Encoder
Ground Truth
```

가 동일 시간축에서 비교 가능해야 한다.

---

### Phase 3 — Baseline EKF

1. Fixed EKF 구성
2. 정상 주행 성능 측정
3. Wheel Slip 등의 Fault Injection
4. Fixed EKF 성능 기록

**완료 조건**

Baseline ATE/RPE/RMSE 확보.

---

### Phase 4 — Fault Injection

1. Encoder Bias
2. Encoder Drift
3. Encoder Wheel Slip
4. IMU Bias
5. LiDAR degradation
6. 복합 fault

**완료 조건**

각 fault의 발생 시간과 강도를 정확하게 기록할 수 있어야 한다.

---

### Phase 5 — Shadow EKF

1. Fixed Shadow EKF 별도 구성
2. Innovation 계산
3. NIS 계산
4. 정상/이상 상태의 통계 확인

**완료 조건**

Adaptive EKF의 변화가 AI 입력 Feature에 영향을 주지 않는 독립적인 pipeline 확보.

---

### Phase 6 — Feature Dataset 생성

1. Sensor residual 생성
2. Innovation 생성
3. NIS 생성
4. Temporal statistics 생성
5. Stationary flag 추가
6. Ground Truth 기반 Reliability Label 생성

**완료 조건**

```text
Features → [s_v, s_ω]
```

형태의 학습 데이터셋 완성.

---

### Phase 7 — AI 모델 학습

순서:

```text
Rule Baseline
↓
Isolation Forest
↓
MLP Regression
↓
Autoencoder 비교
```

평가 기준:

- F1
- AUROC
- Detection Latency
- False Alarm
- Inference Time

**완료 조건**

최종 모델 선정 근거 확보.

---

### Phase 8 — Reliability Smoothing

1. Raw reliability 출력 확인
2. Fast Attack 적용
3. Recovery Gate 적용
4. Stationary 예외 처리
5. 채터링 측정

**완료 조건**

Fault 발생 시 빠르게 하락하고 Fault 종료 후 과도한 지연 없이 안정적으로 회복.

---

### Phase 9 — Adaptive Covariance

1. `s_v` → `R_v` 변환
2. `s_ω` → `R_ω` 변환
3. `R_min`, `R_max` 적용
4. measurement rejection 정책 구현
5. Fixed EKF와 결과 비교

**완료 조건**

AI Reliability 변화가 실제 EKF 동작 및 최종 추정 결과에 반영됨을 증명.

---

### Phase 10 — 최종 검증

비교:

```text
Fixed EKF
vs
Rule Adaptive EKF
vs
AI Adaptive EKF
```

시나리오:

- 정상
- Wheel Slip
- Encoder Drift
- IMU Bias
- LiDAR degradation
- 복합 fault
- Stationary

평가:

- ATE
- RPE
- RMSE
- F1
- Detection Latency
- Recovery Time
- False Alarm Rate

---

### Phase 11 — 시각화 및 해커톤 데모

화면에 최소한 다음 정보를 보여준다.

```text
┌─────────────────────────────────────┐
│      Adaptive Sensor Reliability    │
├─────────────────────────────────────┤
│ LiDAR                                │
│  Reliability : 0.93                  │
│                                      │
│ IMU                                  │
│  Reliability : 0.91                  │
│                                      │
│ Encoder                              │
│  v Reliability : 0.22                │
│  ω Reliability : 0.90                │
│                                      │
│ State: DEGRADED                      │
│ Encoder covariance ↑                 │
└─────────────────────────────────────┘
```

그리고 아래 그래프를 함께 표시한다.

- Ground Truth trajectory
- Fixed EKF trajectory
- Adaptive EKF trajectory
- `s_v` / `s_ω`
- 센서 residual
- NIS
- covariance 변화

---

## 24. 최종 데모 시나리오

### Step 1 — 정상

```text
LiDAR      0.94
IMU        0.92
Encoder v  0.95
Encoder ω  0.93
```

Fixed/Adaptive EKF 모두 정상 추정.

### Step 2 — Wheel Slip 주입

Encoder의 선속도가 실제 이동보다 크게 나타난다.

```text
Encoder v ↑
LiDAR v   정상
IMU       정상
```

AI:

```text
s_v_encoder : 0.95 → 0.72 → 0.43 → 0.20
s_ω_encoder : 0.93 → 0.91 → 0.90 → 0.90
s_v_lidar   : 0.94 → 0.94 → 0.93 → 0.94   (영향 없음 — 센서 분리 설계의 핵심 확인 포인트)
```

### Step 3 — Covariance 변화

```text
R_v ↑
R_ω 유지
```

따라서 선속도 측정의 영향은 감소하지만 회전 정보는 계속 활용한다.

### Step 4 — 추정 결과 비교

```text
Fixed EKF
→ 오차 증가

AI Adaptive EKF
→ 오차 증가 억제
```

### Step 5 — Slip 종료

Residual/NIS/consistency가 정상으로 돌아온 뒤 Reliability가 점진적으로 회복된다.

### Step 6 — 최종 결과

화면에 다음을 보여준다.

```text
Fault detected
       ↓
Reliability decreased
       ↓
Covariance adapted
       ↓
Sensor influence reduced
       ↓
State estimation stabilized
```

---

## 25. 예상 문제와 대응

### 문제 1 — 두 센서가 동시에 틀린 경우

대응:

- Shadow EKF Innovation/NIS 활용
- temporal consistency 활용
- 모든 센서가 불확실하면 특정 센서를 강제로 정상 판정하지 않음
- uncertainty 증가 상태를 허용

### 문제 2 — 센서가 순간적으로 튀는 경우

대응:

- temporal window
- asymmetric smoothing
- NIS 기반 통계적 확인

### 문제 3 — Recovery가 늦어지는 경우

대응:

- 단순 시간 기반 release 대신 Recovery Gate 적용
- `residual + NIS + consistency`를 함께 확인

### 문제 4 — 공분산 폭증

대응:

- bounded covariance inflation
- `R_max` 설정
- 지나치게 작은 Reliability 직접 나눗셈 금지

### 문제 5 — Sim-to-Real 차이

대응:

- 정상 Gazebo noise 다양화
- Fault 강도와 형태 다양화
- 실제 센서 데이터가 가능하면 별도 validation set 구성
- 특정 fault signature 암기 여부 점검

### 문제 6 — 정지 상태 False Alarm

대응:

- stationary detector
- Zero-Velocity 처리
- 저속 encoder quantization 고려
- IMU noise와 fault 구분

---

## 26. 취업 포트폴리오에서 강조할 기술

### AI / ML

- Anomaly Detection
- MLP Regression
- Autoencoder
- Time-Series Feature Engineering
- Statistical Gating
- Model Evaluation

### Robotics

- LiDAR Odometry
- IMU
- Wheel Encoder
- EKF
- Sensor Fusion
- Fault Injection

### System Engineering

- ROS2
- rosbag2
- Time Synchronization
- Dynamic Covariance
- Numerical Stability
- Real-Time Inference
- Reliability Monitoring

### 핵심 포트폴리오 문장

> 다중 센서의 시계열 일관성과 Shadow EKF의 Innovation/NIS를 기반으로 센서별·자유도별 Reliability를 AI로 추정하고, 이를 EKF Measurement Covariance에 동적으로 반영하여 Wheel Slip 및 센서 이상 상황에서 상태 추정 오차를 감소시키는 Adaptive Sensor Reliability 시스템을 개발하였다.

---

## 27. 최종 기술 목표

최종 목표는 단순한 **Sensor Fault Detection**이 아니다.

```text
Sensor Fault Detection
        ↓
Sensor Reliability Estimation
        ↓
DOF-level Reliability
        ↓
Adaptive Covariance
        ↓
Robust EKF State Estimation
```

즉 다음 질문에 답하는 시스템을 만드는 것이 목표다.

> **"지금 이 센서가 고장났는가?"가 아니라, "지금 이 센서의 이 관측을 얼마나 믿어야 하는가?"**

---

## 28. 최종 개발 범위 요약

### 반드시 구현

- LiDAR Odometry
- IMU / Encoder preprocessing
- Time Synchronization
- Fixed Shadow EKF
- Innovation / NIS
- Sensor residual features
- Ground Truth 기반 training label
- AI Reliability model
- `s_v`, `s_ω` 분리
- Reliability smoothing
- Bounded covariance scaling
- Adaptive EKF
- Fault Injection
- Fixed vs Rule vs AI 비교실험
- ATE/RPE/RMSE 평가
- ONNX Runtime (배포 시 CPU 추론용으로 필수)

### 선택 구현

- Autoencoder 비교
- C++ inference node
- Foxglove / PlotJuggler 실시간 시각화
- 실제 ROS2 로봇 데이터 validation

### 구현하지 않음

- Nav2
- SLAM 자체 개발
- Path Planning
- Camera
- LLM

---

## 29. 최종 기대 결과

본 프로젝트가 성공적으로 완료되면 다음을 실험적으로 보여주는 것을 목표로 한다.

1. 정상 상태에서는 Fixed EKF와 유사한 추정 성능을 유지한다.
2. Wheel Slip 발생 시 Encoder의 선속도 Reliability만 낮아진다.
3. 정상적인 회전 정보는 불필요하게 폐기되지 않는다.
4. AI Reliability에 따라 EKF covariance가 동적으로 변화한다.
5. Fixed EKF 대비 Fault 상황의 ATE/RPE가 감소한다.
6. Rule-based Adaptive EKF 대비 AI 방식이 미세한 이상 또는 복합 상황에서 추가적인 이점을 보인다.
7. Reliability가 순간적인 noise에 의해 채터링하지 않는다.
8. Fault 종료 후 Recovery가 안정적으로 수행된다.
9. 센서 전부가 불확실한 상황에서는 임의의 정답을 생성하지 않고 불확실성을 유지한다.

---

## 30. 핵심 한계 및 향후 개선

### 현재 한계

- Gazebo Fault Injection이 실제 고장 상황을 완전히 대체할 수는 없다.
- Ground Truth 기반 Label은 실제 환경의 절대적인 정답이 아니다.
- LiDAR Scan Matching 성능이 환경에 따라 달라질 수 있다.
- AI Reliability와 EKF covariance의 관계는 실험적 calibration이 필요하다.

### 향후 개선

- 실제 로봇 센서 데이터 기반 validation
- 환경별 Reliability calibration
- 센서별/DOF별 Bayesian uncertainty modeling
- Adaptive process noise `Q` 추정
- 다양한 로봇 플랫폼으로 일반화
- Edge device용 ONNX Runtime 최적화

---

## 31. 배포 및 패키징 전략

본 프로젝트는 실험/데모로 끝나지 않고, 실제 ROS2 로봇을 사용하는 다른 사용자들이 가져다 쓸 수 있는 형태로 배포하는 것을 목표로 한다.

### 31.1 핵심 설계 전환 — robot_localization 호환 Covariance Injector

자체 EKF를 새로 만들어 배포하는 대신, 이미 널리 쓰이는 `robot_localization`의 `ekf_node`에 그대로 연결할 수 있는 구조로 패키징한다.

`robot_localization`의 `ekf_node`는 입력 메시지(`nav_msgs/Odometry`, `sensor_msgs/Imu`, `geometry_msgs/TwistWithCovarianceStamped`)에 실려 있는 covariance 값을 그대로 measurement covariance(R)로 사용한다. 따라서 원본 센서 토픽을 EKF에 직접 연결하는 대신, 그 사이에 Covariance Injector 노드를 두어 "동일한 메시지 타입, covariance만 Reliability에 따라 조정된 버전"을 republish한다.

```text
원본 센서 토픽
      ↓
Covariance Injector (본 프로젝트)
      ↓
동일 메시지 + 조정된 covariance
      ↓
사용자의 기존 ekf_filter_node (robot_localization)
```

사용자 입장에서 필요한 변경은 다음 한 줄로 축소된다.

```yaml
odom0: /wheel/odom/adaptive   # 기존: /wheel/odom
```

새 EKF를 학습하거나 기존 파이프라인을 교체할 필요가 없어 채택 장벽을 낮춘다.

### 31.2 패키지 구조

```text
adaptive_sensor_reliability/
├── asr_msgs/                       # ament_cmake + rosidl
│   └── msg/SensorReliability.msg
├── asr_shadow_ekf/                 # 고정 R, Innovation/NIS만 출력
├── asr_reliability_estimator/      # feature builder + AI 추론 + smoothing
│   ├── model/reliability.onnx
│   └── asr_reliability_estimator/
│       ├── feature_builder.py
│       ├── reliability_node.py
│       └── covariance_injector.py
├── asr_bringup/                    # launch + 데모 config/world
└── asr_evaluation/                 # ATE/RPE/RMSE, rosbag 벤치마크 스크립트
```

`SensorReliability.msg`:

```text
std_msgs/Header header
string sensor_name
float64 s_v
float64 s_omega
```

### 31.3 모델 배포 — ONNX Runtime

학습된 모델은 PyTorch 형태로 배포하지 않고 ONNX로 export하여 `asr_reliability_estimator/model/`에 포함한다. 사용자 로봇에 PyTorch/CUDA 설치를 요구하면 채택률이 크게 떨어지므로, CPU 전용 ONNX Runtime으로 추론하여 의존성을 최소화한다. 28절에서 "선택 구현"이었던 ONNX Runtime은 배포를 전제로 "반드시 구현"으로 격상한다.

### 31.4 재현성 — Dockerfile

Gazebo 시뮬레이션, Fault Injection, 학습 파이프라인까지 포함하면 의존성이 무거워 README만으로는 재현이 어렵다. 용도가 다른 두 종류의 이미지를 분리해서 제공한다.

```text
docker/
├── Dockerfile.runtime   # ROS2 + robot_localization + ONNX Runtime (경량, 실제 로봇 배포용)
└── Dockerfile.dev       # + Gazebo + PyTorch + 학습 파이프라인 (재현/재학습용)
```

단순히 사용해보려는 사용자는 `Dockerfile.runtime`만 받으면 되고, 재현·재학습이 필요한 사용자는 `Dockerfile.dev`를 사용한다.

### 31.5 타겟 배포판

2026년 9월 기준 ROS2 배포판 현황: Humble은 2027년 5월까지, Jazzy는 2029년까지 LTS가 유지되고, 2026년 5월 출시된 Lyrical Luth는 2031년까지 지원되는 최신 LTS이다.

- 본 wheelchair 프로젝트는 Humble을 그대로 유지한다.
- 배포용 패키지는 Jazzy를 기본 타겟으로 하고, CI에서 Humble도 함께 빌드·테스트한다.
- Lyrical은 출시 초기이므로 초기 지원 대상에서 제외한다.

### 31.6 배포 로드맵

#### 1단계 — GitHub 공개

- README (데모 GIF, Fixed vs Adaptive EKF ATE 비교 그래프)
- Apache-2.0 라이선스
- `package.xml` / rosdep 의존성 명시
- GitHub Actions: Humble + Jazzy 이미지에서 colcon build/test

#### 2단계 — 데모 온보딩

- Gazebo world + 예제 rosbag 동봉
- `ros2 launch asr_bringup asr_demo.launch.py`로 자신의 로봇 없이도 즉시 재현 가능하게 구성

#### 3단계 — 검색 노출

- `rosdistro`에 문서/소스 인덱스용 PR 제출 (바이너리 릴리즈 불필요)
- ROS Index(index.ros.org) 등록으로 검색 노출 확보

#### 4단계 — 정식 배포

- `bloom-release`로 rosdistro에 정식 등록
- `apt install ros-jazzy-adaptive-sensor-reliability` 지원
- 비교실험 완료 및 실제 로봇 검증 이후 진행

4단계(정식 배포)까지를 최종 목표로 한다.

---

# 결론

**Adaptive Sensor Reliability AI**는 AI가 EKF를 대체하는 프로젝트가 아니다.

핵심 구조는 다음과 같다.

```text
LiDAR + IMU + Encoder
          ↓
   Sensor Relationship
          ↓
   Shadow EKF Innovation/NIS
          ↓
      Reliability AI
          ↓
      s_v / s_ω
          ↓
  Bounded Covariance Scaling
          ↓
      Adaptive EKF
          ↓
    Stable Fused State
```

이 구조를 통해 기존의 고정된 센서 신뢰도 설정에서 벗어나 **현재 상황에 따라 센서 관측의 영향력을 동적으로 조절하는 상태 추정 시스템**을 구현한다.

특히 `Fixed EKF → Rule-based Adaptive EKF → AI Adaptive EKF`의 단계별 비교를 통해 AI 사용의 필요성을 정량적으로 검증하는 것을 본 프로젝트의 핵심 실험으로 한다.
