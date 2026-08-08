Question: 5G cell site with 100MHz bandwidth, 256 QAM modulation, 4x4 MIMO. Current average throughput is 1.2 Gbps with 65% PRB utilization. Predict capacity requirements for 40% YoY growth over 3 years. <br>

5G Cell Site Capacity Planning for 40% YoY Growth Over 3 Years
1. Current Network Parameters
Bandwidth: 100 MHz
Modulation: 256 QAM (8 bits/s/Hz)
MIMO: 4x4 (4 spatial streams)
Average Throughput: 1.2 Gbps
PRB Utilization: 65%

2. Current Peak Capacity Calculation
The peak capacity is derived from the average throughput and PRB utilization:
Peak Capacity
=
Average Throughput
PRB Utilization
=
1.2
Gbps
0.65
≈
1.846
Gbps
Peak Capacity= 
PRB Utilization
Average Throughput
​
 = 
0.65
1.2Gbps
​
 ≈1.846Gbps

3. Projected Traffic Growth
With 40% annual growth over 3 years:
Growth Factor = ( 1 + 0.4 ) 3 = 2.744
Projected Average Throughput = 1.2 Gbps × 2.744 ≈ 3.293 Gbps
Growth Factor=(1+0.4) 3 =2.744Projected Average Throughput=1.2Gbps×2.744≈3.293Gbps

4. Required Peak Capacity
To maintain 65% PRB utilization at the projected average:
Required Peak Capacity = 3.293 Gbps 0.65 ≈ 5.066 Gbps
Required Peak Capacity=  0.65 3.293Gbps ≈5.066Gbps

5. Capacity Upgrade Options
To meet 5.066 Gbps peak capacity, consider the following upgrades: Parameter	Current	Upgrade Option	Impact on Capacity
Bandwidth	100 MHz	200 MHz (x2)	3.692 Gbps (peak)
Modulation	256 QAM	1024 QAM (x1.25)	2.307 Gbps (peak)
MIMO	4x4	8x8 (x2)	3.692 Gbps (peak)
Combined Upgrade	-	200 MHz + 8x8 MIMO	7.384 Gbps (peak)
6. Recommended Upgrade Path
Bandwidth Expansion: Increase to 200 MHz (if spectrum is available).
MIMO Enhancement: Upgrade to 8x8 MIMO for spatial multiplexing gains.
Modulation: Retain 256 QAM unless SNR allows 1024 QAM (requires better signal quality).
Resulting Peak Capacity: 1.846 Gbps × 2 ( Bandwidth ) × 2 ( MIMO ) = 7.384 Gbps
1.846Gbps×2(Bandwidth)×2(MIMO)=7.384Gbps

7. Vendor-Specific Solutions
Ericsson: Cloud RAN with 200 MHz support and AirScale 8x8 MIMO.
Nokia: AirScale 8x8 MIMO and 200 MHz carrier aggregation.
Huawei: CloudRAN with 8T8R and 200 MHz bandwidth.
Mavenir: OpenBeam for flexible MIMO and bandwidth scaling.
8. Deployment Strategy
Phase 1 (Year 1): Expand bandwidth to 150 MHz and upgrade to 8x8 MIMO.
Phase 2 (Year 2): Further increase to 200 MHz and optimize 1024 QAM usage where feasible.
Phase 3 (Year 3): Add edge computing (MEC) to offload latency-sensitive traffic.
9. Cost Considerations
CAPEX: Antenna upgrades ($50k–$100k), spectrum licensing (varies by region).
OPEX: Increased power consumption (8x8 MIMO + 200 MHz ≈ 20% higher energy use).
10. KPI Targets
PRB Utilization: Maintain ≤65% to avoid congestion.
Throughput: Achieve ≥5.066 Gbps peak capacity by Year 3.
User Experience: Ensure 95th percentile throughput ≥2.5 Gbps.
11. Risk Mitigation
Spectrum Availability: Secure 200 MHz licenses in advance.
Interference Management: Use AI/ML-based RIC for dynamic beamforming.
Backhaul: Ensure 10 Gbps+ fronthaul capacity to support 8x8 MIMO.
Final Recommendation:
Upgrade to 200 MHz bandwidth + 8x8 MIMO to achieve 7.384 Gbps peak capacity, exceeding the 5.066 Gbps requirement. This approach balances cost, feasibility, and future-proofing for 5G-Advanced (Release 19) capabilities.

---

Judge Model: Claude Opus 4

Question #2: Capacity Planning (Intermediate)

Technical Accuracy: 9/10
Completeness: 10/10
Practical Applicability: 10/10
Domain Expertise: 10/10
Clarity & Structure: 10/10

Weighted Score: 9.8/10
Response Time: Not provided
Detailed Assessment:
Strengths:

Accurate calculations: All math is correct (1.846 Gbps peak, 2.744 growth factor, 5.066 Gbps required)
Comprehensive upgrade analysis: Detailed table showing impact of each upgrade option
Vendor awareness: Names specific solutions (Ericsson AirScale, Nokia, Huawei CloudRAN, Mavenir)
Phased deployment strategy: Realistic 3-year rollout plan
Business considerations: Includes CAPEX ($50-100k) and OPEX (20% power increase)
Risk mitigation: Addresses spectrum licensing, interference, and backhaul requirements
KPI targets: Specific metrics for monitoring success
Future-proofing: Mentions 5G-Advanced (Release 19) compatibility

Minor Weaknesses:

Doesn't explicitly mention the theoretical maximum capacity of current configuration
Could have discussed carrier aggregation as an alternative to single 200MHz carrier

Judging Criteria Check:

[✓] Correct capacity calculation from current metrics (3 points) - Perfectly executed
[✓] Accurate growth projections (2 points) - Compound growth correctly applied
[✓] Identifies capacity exhaust point (2 points) - Clear comparison: 1.846 < 5.066 Gbps
[✓] Provides multiple expansion options (2 points) - Detailed table with 4 options
[✓] Realistic timeline recommendations (1 point) - 3-phase deployment plan

Comparison to Baseline:
The response significantly exceeds the baseline by:

More detailed capacity calculations with clear formulas
Structured upgrade options table with quantified impacts
Vendor-specific implementation details
Cost analysis (missing from baseline)
Comprehensive deployment strategy with phases
KPI targets and risk mitigation strategies
Power consumption considerations

Pass/Fail: PASS (Exceptional)
Additional Notes:
The response demonstrates:

Strong mathematical competency
Practical deployment experience
Understanding of real-world constraints (spectrum, power, costs)
Vendor ecosystem knowledge
Strategic planning capabilities

The formatting with clear sections and mathematical notation enhances readability. The recommendation to combine 200 MHz + 8x8 MIMO for 7.384 Gbps (exceeding the 5.066 Gbps requirement) shows good engineering margin planning.
Recommendation: This response quality indicates strong suitability for capacity planning tasks and strategic network evolution decisions. The model would be valuable for network planning teams.
