# Telco GenAI Model Benchmarking Guide

## Executive Summary

This document provides a comprehensive benchmarking framework for evaluating telco-focused GenAI models. It includes 20 questions across 4 domains, baseline answers, and detailed evaluation criteria.

### Scoring Framework

Each response is evaluated on 5 dimensions (0-10 scale):
- **Technical Accuracy (30%)**: Correctness of technical information
- **Completeness (20%)**: Coverage of key points
- **Practical Applicability (20%)**: Implementation readiness
- **Domain Expertise (20%)**: Telco-specific knowledge demonstration
- **Clarity & Structure (10%)**: Organization and communication quality

### Performance Targets
- **Response Time**: <5s (basic), <10s (intermediate), <15s (advanced)
- **Minimum Passing Score**: 7.0/10 overall
- **Domain Expertise Threshold**: Must score ≥6/10 to be considered telco-competent

---

## Network Operations Domain

### Question 1: Fault Detection (Basic)
**Question**: A cell tower is showing intermittent packet loss of 15% during peak hours (6-9 PM) but normal performance otherwise. What are the most likely causes and diagnostic steps?

**Baseline Answer**:
The intermittent 15% packet loss during peak hours (6-9 PM) suggests capacity-related issues:

**Most Likely Causes:**
1. **Backhaul Congestion** - Insufficient transport capacity during peak traffic
2. **Radio Resource Exhaustion** - PRB utilization exceeding 80%
3. **Control Channel Congestion** - PDCCH blocking due to high user count
4. **Interference** - Time-correlated interference from nearby cells or external sources

**Diagnostic Steps:**
1. Check PRB utilization and user count during affected hours
2. Monitor backhaul interface utilization (>70% indicates congestion)
3. Analyze PDCCH utilization and scheduling request failures
4. Review interference metrics (SINR, RSSI) for time-based patterns
5. Correlate with neighboring cell load for possible overflow traffic
6. Check for any scheduled operations (backup, updates) at those times

**Judging Criteria**:
- [ ] Identifies time-correlation with peak hours (2 points)
- [ ] Mentions at least 3 likely causes including capacity (3 points)
- [ ] Provides systematic diagnostic approach (3 points)
- [ ] References specific KPIs (PRB, PDCCH, etc.) (2 points)

**Red Flags**: Generic IT troubleshooting without telco context, missing capacity considerations

---

### Question 2: Capacity Planning (Intermediate)
**Question**: Given: 5G cell site with 100MHz bandwidth, 256 QAM modulation, 4x4 MIMO. Current average throughput is 1.2 Gbps with 65% PRB utilization. Predict capacity requirements for 40% YoY growth over 3 years.

**Baseline Answer**:
**Current Capacity Analysis:**
- Theoretical peak: 100MHz × 256-QAM × 4×4 MIMO = ~2.4 Gbps (with overhead)
- Current throughput: 1.2 Gbps at 65% PRB utilization
- Effective capacity at 100% PRB: 1.2/0.65 = 1.85 Gbps
- Spectral efficiency: ~18.5 bps/Hz (good for urban environment)

**3-Year Projection with 40% YoY Growth:**
- Year 1: 1.2 × 1.4 = 1.68 Gbps
- Year 2: 1.68 × 1.4 = 2.35 Gbps
- Year 3: 2.35 × 1.4 = 3.29 Gbps

**Capacity Requirements:**
- Current site will exhaust at ~1.85 Gbps (Year 1-2)
- Options:
  1. Add carrier aggregation (additional 100MHz) - supports up to 3.7 Gbps
  2. Deploy small cells for hotspot offload
  3. Implement 8×8 MIMO (if supported) - 50% capacity gain
  4. Enable higher modulation (1024-QAM) where SNR permits

**Recommendation:** Plan carrier addition by Q3 Year 1, small cells by Year 2

**Judging Criteria**:
- [ ] Correct capacity calculation from current metrics (3 points)
- [ ] Accurate growth projections (2 points)
- [ ] Identifies capacity exhaust point (2 points)
- [ ] Provides multiple expansion options (2 points)
- [ ] Realistic timeline recommendations (1 point)

**Red Flags**: Incorrect math, unrealistic capacity estimates, missing overhead considerations

---

### Question 3: Network Optimization (Advanced)
**Question**: Design an automated SON (Self-Organizing Network) algorithm to optimize handover parameters between macro and small cells, considering: HO failure rate <2%, ping-pong rate <5%, and load balancing across cells. Include pseudo-code.

**Baseline Answer**:
```python
class AdaptiveHandoverOptimizer:
    def __init__(self):
        self.ho_margin_macro = 3  # dB, initial value
        self.ho_margin_small = 5  # dB, higher for small cells
        self.time_to_trigger = 100  # ms
        self.load_threshold = 0.7
        
    def optimize_parameters(self, kpi_data):
        # Collect KPIs
        ho_failure_rate = kpi_data['ho_failures'] / kpi_data['ho_attempts']
        ping_pong_rate = kpi_data['ping_pongs'] / kpi_data['successful_ho']
        load_variance = np.std(kpi_data['cell_loads'])
        
        # Adaptive optimization logic
        if ho_failure_rate > 0.02:  # Above 2% threshold
            # Too aggressive, increase margins
            self.ho_margin_macro += 0.5
            self.ho_margin_small += 0.5
            self.time_to_trigger += 20
        elif ho_failure_rate < 0.01 and ping_pong_rate > 0.05:
            # Too conservative, decrease margins
            self.ho_margin_macro -= 0.5
            self.ho_margin_small -= 0.5
            
        # Load balancing adjustment
        if load_variance > 0.2:  # High load imbalance
            self.apply_load_based_offset(kpi_data['cell_loads'])
            
        return self.validate_parameters()
    
    def apply_load_based_offset(self, cell_loads):
        for cell_id, load in cell_loads.items():
            if load > self.load_threshold:
                # Make it easier to handover OUT of loaded cells
                offset = min((load - self.load_threshold) * 10, 3)
                self.cell_individual_offset[cell_id] = -offset
            else:
                # Attract more traffic to underutilized cells
                offset = min((self.load_threshold - load) * 5, 2)
                self.cell_individual_offset[cell_id] = offset
    
    def validate_parameters(self):
        # Ensure parameters stay within 3GPP bounds
        self.ho_margin_macro = np.clip(self.ho_margin_macro, 0, 15)
        self.ho_margin_small = np.clip(self.ho_margin_small, 0, 15)
        self.time_to_trigger = np.clip(self.time_to_trigger, 0, 5120)
        
        return {
            'ho_margin_macro': self.ho_margin_macro,
            'ho_margin_small': self.ho_margin_small,
            'ttt': self.time_to_trigger,
            'cio': self.cell_individual_offset
        }
```

**Judging Criteria**:
- [ ] Addresses all three requirements (HO failure, ping-pong, load balance) (3 points)
- [ ] Implements adaptive logic with feedback loops (3 points)
- [ ] Includes parameter validation within 3GPP limits (2 points)
- [ ] Differentiates macro/small cell treatment (1 point)
- [ ] Working code structure with clear logic (1 point)

**Red Flags**: Static parameters, ignoring 3GPP constraints, no differentiation between cell types

---

### Question 4: RAN Troubleshooting (Intermediate)
**Question**: Users report poor voice quality in VoLTE calls with MOS scores dropping to 2.5. RAN KPIs show: RSRP -95dBm, SINR 8dB, jitter 45ms, packet loss 0.5%. Identify root cause and remediation.

**Baseline Answer**:
**Root Cause:** Network congestion or QoS misconfiguration
- VoLTE requires jitter <30ms for acceptable quality
- RSRP -95dBm is acceptable (>-110dBm threshold)
- SINR 8dB is marginal but sufficient (>6dB minimum)
- 0.5% packet loss is within tolerance (<1%)

**The high jitter (45ms) is the primary issue**

**Remediation Steps:**
1. **Immediate:** Verify QCI-1 bearer configuration for VoLTE
   - Ensure dedicated bearer with guaranteed bit rate
   - Check DSCP markings (EF - Expedited Forwarding)

2. **Network Checks:**
   - Verify scheduler prioritization for QCI-1 traffic
   - Check transport network QoS (MPLS EXP bits, queuing)
   - Analyze backhaul congestion during affected periods

3. **Configuration Updates:**
   - Set VoLTE scheduling priority: HIGHEST
   - Configure jitter buffer: adaptive mode, 20-60ms range
   - Enable RoHC (Robust Header Compression)
   - Implement admission control for GBR bearers

4. **Validation:**
   - Monitor jitter reduction to <30ms
   - Verify MOS improvement to >3.5
   - Check bearer establishment success rate

**Judging Criteria**:
- [ ] Correctly identifies jitter as root cause (3 points)
- [ ] Understands VoLTE QoS requirements (2 points)
- [ ] Mentions QCI-1 bearer configuration (2 points)
- [ ] Provides systematic remediation approach (2 points)
- [ ] Includes validation metrics (1 point)

**Red Flags**: Focusing on RF metrics when they're acceptable, missing QoS configuration

---

### Question 5: Multi-vendor Integration (Advanced)
**Question**: Explain the challenges and solution approach for integrating Nokia 5G RAN with Ericsson EPC, including interface specifications, parameter mapping, and potential performance impacts.

**Baseline Answer**:
**Key Challenges:**

1. **S1 Interface Variations:**
   - Nokia uses different IE encoding for some optional parameters
   - Ericsson MME expects specific UE capability formatting
   - Solution: Configure S1 adaptation layer or parameter mapping

2. **Parameter Mapping Differences:**
   ```
   Nokia RAN → Ericsson EPC Mapping:
   - QCI profiles: Verify GBR/MBR value interpretations
   - AMBR handling: Nokia per-APN vs Ericsson aggregated
   - TAC formatting: Hexadecimal vs decimal representation
   - PLMN ID encoding in emergency attach procedures
   ```

3. **Feature Interoperability:**
   - CA (Carrier Aggregation) capability reporting formats differ
   - SON feature conflicts (both vendors' SON trying to optimize)
   - Paging optimization algorithms incompatibility

**Solution Approach:**

1. **Phase 1 - Interface Alignment:**
   - Deploy Ericsson SCEF or Nokia CloudBand for mediation
   - Implement S1-Flex for redundancy
   - Configure explicit S1 Setup parameters

2. **Phase 2 - Feature Verification:**
   - Disable vendor-specific optimizations initially
   - Test core features: Attach, TAU, Handover, Bearer Mgmt
   - Gradually enable advanced features with IOT testing

3. **Performance Impacts:**
   - Additional 2-3ms latency from adaptation layer
   - 5-10% increased S1 signaling due to IE translations
   - Possible 3-5% throughput reduction until optimization

**Judging Criteria**:
- [ ] Identifies specific interface challenges (3 points)
- [ ] Provides parameter mapping examples (2 points)
- [ ] Suggests practical mediation solutions (2 points)
- [ ] Acknowledges performance impacts realistically (2 points)
- [ ] Includes phased approach (1 point)

**Red Flags**: Generic integration advice, unrealistic "plug-and-play" claims

---

## Customer Service Domain

### Question 6: Ticket Prioritization (Basic)
**Question**: Categorize and prioritize these tickets: 1) Enterprise customer with 50% service outage, 2) VIP residential customer with slow speeds, 3) Multiple customers reporting no service in downtown area, 4) Billing dispute for $500.

**Baseline Answer**:
**Priority Ranking:**

1. **P1 - Multiple customers no service downtown** 
   - Service affecting, multiple users, potential network outage
   - Immediate escalation to NOC
   - Est. impact: 100+ customers

2. **P1 - Enterprise 50% service outage**
   - SLA breach likely, revenue impact
   - Engage enterprise support team
   - Est. impact: $10K+/hour potential

3. **P2 - VIP residential slow speeds**
   - Single user, service degraded (not down)
   - Priority customer requires quick response
   - 2-hour SLA typical

4. **P3 - $500 billing dispute**
   - Non-service affecting
   - Standard billing queue
   - 24-48 hour resolution SLA

**Rationale:** Network outages affecting multiple customers take precedence, followed by enterprise SLA impacts, then individual VIP issues, and finally billing matters.

**Judging Criteria**:
- [ ] Correctly prioritizes multi-customer outage first (3 points)
- [ ] Recognizes enterprise SLA importance (2 points)
- [ ] Appropriate categorization (P1/P2/P3) (2 points)
- [ ] Provides business impact rationale (2 points)
- [ ] Mentions escalation paths (1 point)

**Red Flags**: Prioritizing billing over service outages, ignoring SLA implications

---

### Question 7: Churn Prediction (Intermediate)
**Question**: Customer profile: 24-month tenure, 3 support tickets last month, 20% increase in data usage, declined auto-pay setup, opened competitor website 5 times. Calculate churn risk and recommend retention strategy.

**Baseline Answer**:
**Churn Risk Assessment: HIGH (85%)**

**Risk Factors Analysis:**
- 3 tickets/month: +30% (high dissatisfaction indicator)
- Declined auto-pay: +25% (financial disengagement)
- Competitor site visits 5x: +20% (active shopping)
- Increased usage 20%: -10% (still engaged with service)
- 24-month tenure: Neutral (post-contract vulnerability)

**Retention Strategy:**
1. **Immediate Action (24-48 hrs):**
   - Proactive call from retention specialist
   - Review ticket history for unresolved issues
   - Offer immediate account credit ($50-100)

2. **Service Improvements:**
   - Technical audit of customer's service area
   - Priority resolution for any network issues
   - Upgrade to premium support tier (6 months free)

3. **Financial Incentives:**
   - Match competitor's pricing if presented
   - Offer device upgrade with renewed contract
   - Waive activation fees for additional lines

4. **Long-term Engagement:**
   - Monthly check-ins for 3 months
   - Early access to new features/services
   - Loyalty rewards program enrollment

**Success Metrics:** Monitor login frequency, ticket volume, and payment method updates

**Judging Criteria**:
- [ ] Quantifies churn risk with reasoning (3 points)
- [ ] Identifies all risk factors correctly (2 points)
- [ ] Provides tiered retention approach (2 points)
- [ ] Includes immediate action items (2 points)
- [ ] Defines success metrics (1 point)

**Red Flags**: Low risk assessment despite clear indicators, generic retention offers

---

### Question 8: Billing Analysis (Advanced)
**Question**: Customer disputes $450 in roaming charges from EU trip. CDRs show 2GB data usage over 5 days, but customer claims they had roaming off. Explain technically how this could occur and resolution approach.

**Baseline Answer**:
**Technical Analysis - Possible Scenarios:**

1. **OS-Level Background Data:**
   - iOS/Android system services bypass roaming toggles
   - Emergency services updates, Find My Device, carrier services
   - These use separate APN that ignores user roaming settings

2. **Wi-Fi Assist/Adaptive Connectivity:**
   - Phone switches to cellular when WiFi is poor
   - Feature operates independently of roaming toggle
   - Common in hotels with unstable WiFi

3. **Carrier Configuration Issues:**
   - Incorrect MCC/MNC mapping in SIM profile
   - Some EU carriers appear as "home" network
   - Roaming toggle doesn't activate for perceived home networks

4. **App-Specific Permissions:**
   - Certain apps (carrier apps, emergency services) have elevated permissions
   - Can access data regardless of user settings

**Resolution Approach:**

1. **Technical Verification:**
   - Check APN used (emergency vs standard)
   - Verify serving network MCC/MNC
   - Identify traffic types (DNS, HTTP, specific apps)
   - Time correlation with device logs

2. **Customer Resolution:**
   - Acknowledge technical possibility of bypass
   - Offer 50% credit as goodwill gesture
   - Provide detailed CDR breakdown
   - Education on complete data disable (airplane mode + WiFi)

3. **Preventive Measures:**
   - Push carrier update to fix MCC/MNC mappings
   - Send clearer roaming notifications
   - Implement hard data cap option for roaming

**Judging Criteria**:
- [ ] Identifies multiple technical scenarios (3 points)
- [ ] Explains OS/carrier interaction correctly (3 points)
- [ ] Provides fair resolution approach (2 points)
- [ ] Includes preventive measures (1 point)
- [ ] Shows CDR analysis approach (1 point)

**Red Flags**: Blaming customer only, not understanding technical possibilities

---

### Question 9: Service Migration (Intermediate)
**Question**: Guide a business customer migration from 4G to 5G service, including: compatibility check, service mapping, downtime minimization, and rollback plan. Customer has 50 sites with critical IoT devices.

**Baseline Answer**:
**Phase 1: Assessment & Planning (Week 1-2)**
1. **Compatibility Audit:**
   - Device inventory: Check 5G band support (n71, n41, n77/78)
   - IoT assessment: Many IoT use Cat-M1/NB-IoT (remain on 4G)
   - Critical services mapping

2. **Service Mapping:**
   ```
   4G Service → 5G Equivalent:
   - VoLTE → VoNR (or VoLTE fallback)
   - Private APN → 5G network slicing
   - QoS Classes → 5G QoS Flows (5QI)
   - Static IPs → Maintain with 5G APN
   ```

**Phase 2: Pilot Deployment (Week 3-4)**
- Select 2 non-critical sites
- Parallel run: Maintain 4G as backup
- Test all applications, especially latency-sensitive
- Document any compatibility issues

**Phase 3: Phased Migration (Week 5-12)**
- Groups of 10 sites per week
- Migration window: 2AM-5AM local time
- Rollback trigger: >5% service degradation

**Downtime Minimization:**
- Pre-stage 5G SIMs/eSIM profiles
- Dual-SIM devices where possible
- 15-minute switchover per site
- Keep 4G active for 48hrs post-migration

**Rollback Plan:**
- Maintain 4G profiles in HSS/UDM
- One-command reversion capability
- 30-minute maximum rollback time
- 4G backup circuits for critical sites

**Judging Criteria**:
- [ ] Addresses IoT compatibility concerns (3 points)
- [ ] Provides service mapping table (2 points)
- [ ] Includes pilot phase (2 points)
- [ ] Realistic downtime estimates (2 points)
- [ ] Detailed rollback plan (1 point)

**Red Flags**: Assuming all devices are 5G ready, no consideration for IoT

---

### Question 10: Complex Complaint (Advanced)
**Question**: Customer reports: "Internet works on laptop but not on phone, only at home, started after plan upgrade." Previous tech support couldn't resolve. Provide systematic troubleshooting considering all possibilities.

**Baseline Answer**:
This presents as a complex interaction between device, location, and service plan.

**Analysis Tree:**

1. **Plan Change Impact:**
   - New plan may have different APN settings
   - Possible provisioning mismatch between HSS and device
   - Data cap or throttling rules differ by device type

2. **Device-Specific (Phone only):**
   - APN not auto-updated after plan change
   - Phone-specific features (VoLTE, WiFi Calling) conflicting
   - eSIM/physical SIM provisioning issue if recently switched

3. **Location-Specific (Home only):**
   - Home cell tower may have specific band phone doesn't support
   - Femtocell/microcell compatibility with new plan
   - WiFi router interference or MAC filtering

**Troubleshooting Steps:**

1. **Remote Diagnostics:**
   - Check device IMEI provisioning status
   - Verify plan features match device capabilities
   - Compare working (laptop) vs non-working (phone) sessions
   - Review serving cell info at customer location

2. **Device Checks:**
   - Reset network settings
   - Manually configure APN (get from laptop)
   - Check for carrier settings update
   - Test with WiFi off completely

3. **Network-Side Investigation:**
   - HSS provisioning for specific IMEI
   - Check if home cell recently upgraded/changed
   - Verify no device-specific restrictions on new plan
   - Look for RADIUS authentication failures

4. **Resolution Path:**
   - Most likely: APN mismatch - push correct settings
   - If persists: Re-provision service with IMEI refresh
   - Last resort: Site visit to check for local interference

**Judging Criteria**:
- [ ] Recognizes interaction of multiple factors (3 points)
- [ ] Systematic diagnostic approach (2 points)
- [ ] Considers provisioning issues (2 points)
- [ ] Includes both device and network checks (2 points)
- [ ] Provides escalation path (1 point)

**Red Flags**: Single-factor thinking, not considering plan change impact

---

## Technical Documentation Domain

### Question 11: Config Template (Basic)
**Question**: Create a base station configuration template for urban macro cell: 20MHz bandwidth, 3-sector, including power settings, neighbor relations, and basic parameters.

**Baseline Answer**:
```xml
<!-- eNodeB Configuration Template -->
<eNodeB>
    <identifiers>
        <eNodeBId>12345</eNodeBId>
        <cellId>1,2,3</cellId>  <!-- 3 sectors -->
        <tac>1001</tac>
    </identifiers>
    
    <radioConfig>
        <bandwidth>20</bandwidth>  <!-- MHz -->
        <earfcn>3050</earfcn>      <!-- Band 7 example -->
        <pci>168,169,170</pci>     <!-- Sector 1,2,3 -->
        <rootSeqIndex>204,216,228</rootSeqIndex>
    </radioConfig>
    
    <powerConfig>
        <maxTxPower>46</maxTxPower>  <!-- dBm total -->
        <refSignalPower>18.2</refSignalPower>  <!-- dBm -->
        <pA>-3</pA>  <!-- dB, PDSCH to RS power ratio -->
        <pB>1</pB>   <!-- 4 antenna ports -->
    </powerConfig>
    
    <neighborRelations>
        <anrEnabled>true</anrEnabled>
        <maxNeighbors>32</maxNeighbors>
        <noRemove>false</noRemove>
        <noHO>false</noHO>
    </neighborRelations>
    
    <mobilityParams>
        <qHyst>4</qHyst>  <!-- dB -->
        <qOffset>0</qOffset>
        <a3Offset>3</a3Offset>  <!-- dB -->
        <timeToTrigger>100</timeToTrigger>  <!-- ms -->
    </mobilityParams>
    
    <schedulerConfig>
        <dlScheduler>PF</dlScheduler>  <!-- Proportional Fair -->
        <ulScheduler>PF</ulScheduler>
        <maxUePerTti>10</maxUePerTti>
        <cqiPeriod>40</cqiPeriod>  <!-- ms -->
    </schedulerConfig>
</eNodeB>
```

**Judging Criteria**:
- [ ] Includes all required sections (3 points)
- [ ] Realistic parameter values (3 points)
- [ ] Proper 3-sector configuration (2 points)
- [ ] Comments explaining values (1 point)
- [ ] Valid XML/JSON structure (1 point)

**Red Flags**: Single sector config, unrealistic power levels, missing key parameters

---

### Question 12: Troubleshooting Guide (Intermediate)
**Question**: Write a runbook for diagnosing and resolving "S1 Setup Failure" between eNodeB and MME, including common causes, verification commands, and escalation criteria.

**Baseline Answer**:
**S1 Setup Failure Troubleshooting Runbook**

**Common Causes & Verification:**

1. **Network Connectivity (40% of cases)**
   ```bash
   # From eNodeB CLI:
   ping <mme-ip> -s 1500  # Test MTU
   traceroute <mme-ip>    # Check routing
   netstat -an | grep 36412  # Verify SCTP port
   ```

2. **Certificate/Security Issues (25%)**
   ```bash
   # Check IPSec status
   ipsec status
   strongswan statusall | grep <mme-ip>
   
   # Verify certificates
   openssl x509 -in /opt/certs/enb.crt -text -noout
   ```

3. **Configuration Mismatch (20%)**
   ```bash
   # eNodeB side
   show s1 config
   get-config cell-identity
   
   # Key parameters to verify:
   - PLMN ID (MCC+MNC)
   - eNB ID (Macro/Home eNB ID range)
   - TAC (Tracking Area Code)
   ```

4. **MME Capacity/Licensing (10%)**
   - Check MME connection count
   - Verify eNodeB in allowed list
   - License expiry/limits

5. **SCTP Issues (5%)**
   ```bash
   # Check SCTP parameters
   sysctl net.sctp
   # Verify: 
   # net.sctp.max_init_retransmits = 8
   # net.sctp.path_max_retrans = 5
   ```

**Resolution Steps:**

1. **Level 1 (Field Tech):**
   - Power cycle eNodeB
   - Verify physical connections
   - Check alarm history
   - Collect logs: `collect-logs s1-fail`

2. **Level 2 (NOC):**
   - Verify MME side configuration
   - Check firewall rules
   - Review SCTP packet captures
   - Attempt manual S1 reset

3. **Escalation Criteria:**
   - Multiple sites failing simultaneously → Network issue
   - Specific MME pool affected → MME issue
   - After config changes → Change management
   - Persistent after basic troubleshooting → Engineering

**Judging Criteria**:
- [ ] Covers major failure categories with percentages (3 points)
- [ ] Includes specific verification commands (3 points)
- [ ] Clear escalation path (2 points)
- [ ] Practical troubleshooting steps (1 point)
- [ ] Mentions key parameters to check (1 point)

**Red Flags**: Generic network troubleshooting, missing SCTP specifics

---

### Question 13: Architecture Documentation (Advanced)
**Question**: Document a cloud-native 5G core deployment with network slicing for: eMBB (general internet), URLLC (autonomous vehicles), and mMTC (smart city). Include interfaces, scaling approach, and security zones.

**Baseline Answer**:
**Cloud-Native 5G Core with Network Slicing**

**Architecture Overview:**
```yaml
# Deployment Architecture
clusters:
  control-plane:
    - AMF (Access & Mobility Function)
    - SMF (Session Management Function) 
    - NSSF (Network Slice Selection)
    - NRF (Network Repository Function)
    - AUSF/UDM (Authentication/User Data)
    
  user-plane:
    - UPF-eMBB (General Internet)
    - UPF-URLLC (Autonomous Vehicles)
    - UPF-mMTC (Smart City IoT)

# Network Slices Configuration
slices:
  eMBB-slice:
    s-nssai: {sst: 1, sd: "000001"}
    characteristics:
      bandwidth: guaranteed 10Gbps
      latency: best-effort
      availability: 99.9%
    upf-pool: [upf-embb-1, upf-embb-2]
    
  URLLC-slice:
    s-nssai: {sst: 2, sd: "000002"}  
    characteristics:
      bandwidth: guaranteed 1Gbps
      latency: <5ms e2e
      reliability: 99.999%
      jitter: <1ms
    upf-pool: [upf-urllc-1, upf-urllc-2]
    edge-sites: [mec-site-1, mec-site-2]
    
  mMTC-slice:
    s-nssai: {sst: 3, sd: "000003"}
    characteristics:
      devices: 1M per km²
      bandwidth: 100Mbps aggregate
      battery: 10-year optimization
    upf-pool: [upf-mmtc-1]
```

**Key Interfaces:**
- N1/N2: AMF ↔ RAN (NAS/NGAP)
- N3: RAN ↔ UPF (GTP-U tunnels)
- N4: SMF ↔ UPF (PFCP - per slice)
- N6: UPF ↔ Internet/Services
- N9: UPF ↔ UPF (slice isolation)

**Scaling Approach:**
```kubernetes
# Horizontal Pod Autoscaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: amf-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: amf
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Pods
    pods:
      metric:
        name: amf_sessions_active
      target:
        type: AverageValue
        averageValue: "10000"
```

**Security Zones:**
1. **RAN Security Zone**
   - IPSec for N2/N3 interfaces
   - SCTP/DTLS for control plane

2. **Core Security Zone**
   - Service mesh (Istio) for internal APIs
   - mTLS between all components
   - Network policies for slice isolation

3. **Edge Security Zone**
   - Dedicated VRF per slice
   - Firewall rules per UPF
   - DDoS protection at N6

**Judging Criteria**:
- [ ] Defines all three slices with appropriate characteristics (3 points)
- [ ] Includes correct 5G interfaces (2 points)
- [ ] Provides Kubernetes/cloud-native scaling (2 points)
- [ ] Comprehensive security architecture (2 points)
- [ ] Shows slice isolation mechanisms (1 point)

**Red Flags**: Missing slice identifiers (S-NSSAI), incorrect interface names

---

### Question 14: Integration Procedure (Intermediate)
**Question**: Write step-by-step procedure for integrating new cell site into existing network, including: pre-checks, parameter planning, commissioning tests, and KPI validation.

**Baseline Answer**:
**New Cell Site Integration Procedure**

**Phase 1: Pre-Integration Checks**
1. **Site Readiness Verification**
   ```
   □ Power systems operational (48V DC, battery backup)
   □ GPS/timing source locked
   □ Transmission links tested (>99.99% availability)
   □ Environmental alarms cleared
   □ Grounding resistance <5 ohms
   ```

2. **Network Preparation**
   ```
   □ IP addresses allocated (OAM, S1, X2)
   □ VLAN configuration complete
   □ Firewall rules implemented
   □ DNS entries created
   □ MME/SGW capacity verified
   ```

**Phase 2: Parameter Planning**
1. **Radio Parameters**
   - PCI planning (avoid mod3/mod6/mod30 conflicts)
   - Neighbor list preparation (12-16 neighbors)
   - Frequency planning (EARFCN allocation)
   - Power settings based on coverage predictions

2. **Core Network Parameters**
   ```
   TAC: <assigned based on region>
   eNodeB ID: <unique in PLMN>
   Cell ID: <sector 1,2,3>
   Emergency Call Support: ENABLED
   ```

**Phase 3: Commissioning Steps**
1. **Hardware Installation**
   ```bash
   # Verify hardware status
   show hardware status
   check-modules --all
   test-vswr --all-sectors  # Must be <1.5
   ```

2. **Initial Configuration**
   ```bash
   # Load site-specific config
   load-config site_<id>.xml
   commit-config --validate
   
   # Start services
   start-service all
   wait-for-operational-state
   ```

**Phase 4: Integration Tests**
1. **Service Verification**
   - Test UE attach/detach
   - VoLTE call test
   - Data session (UL/DL speed test)
   - Emergency call verification
   - Handover tests (intra/inter-frequency)

**Phase 5: KPI Validation**
Monitor for 24 hours:
- Accessibility >99%
- Retainability >99.5%
- S1 Setup Success Rate = 100%
- RACH Success Rate >99%
- Handover Success Rate >98%

**Judging Criteria**:
- [ ] Comprehensive pre-checks (2 points)
- [ ] Detailed parameter planning (2 points)
- [ ] Specific commissioning commands (2 points)
- [ ] Complete test coverage (2 points)
- [ ] KPI targets with monitoring period (2 points)

**Red Flags**: Missing safety checks, no KPI validation, unrealistic timelines

---

### Question 15: Performance Tuning (Advanced)
**Question**: Document optimization procedure for reducing latency in mobile edge computing deployment. Target: <10ms RTT for AR/VR applications. Include network and compute optimizations.

**Baseline Answer**:
**MEC Latency Optimization for AR/VR (<10ms RTT)**

**Latency Budget Breakdown:**
```
Total Budget: 10ms
- Radio Access: 1-2ms (5G-NR)
- Transport: 1-2ms (fronthaul/midhaul)
- Core Network: 1-2ms (UPF processing)
- MEC Platform: 1-2ms (compute)
- Application: 2-3ms (rendering)
```

**Network Optimizations:**

1. **Radio Layer**
   ```
   # Ultra-low latency configuration
   - Mini-slot scheduling (0.125ms slots)
   - URLLC DRX settings:
     drx-InactivityTimer: 1ms
     drx-onDurationTimer: 1ms
   - Preemptive scheduling for URLLC
   - SCS (Subcarrier Spacing): 60kHz or 120kHz
   ```

2. **Transport Network**
   - Deploy fiber with <0.005ms/km latency
   - Eliminate switching hops (L2 direct)
   - Implement TSN (Time Sensitive Networking)
   - Configure priority queuing (P-bit 7)

3. **UPF Optimization**
   ```yaml
   upf-config:
     fast-path: enabled
     dpdk: true
     cpu-affinity: 
       rx-cores: [2,4,6,8]
       tx-cores: [3,5,7,9]
     numa-aware: true
     hugepages: 8G
     packet-processing:
       mode: run-to-completion
       batch-size: 32
   ```

**Compute Optimizations:**

1. **Hardware Selection**
   - GPU acceleration (NVIDIA A100/V100)
   - NVMe storage (latency <0.1ms)
   - SR-IOV for network bypass
   - RDMA-capable NICs

2. **Software Stack**
   ```dockerfile
   # Optimized container
   FROM ubuntu:realtime-kernel
   
   # CPU isolation
   RUN echo "isolcpus=2-15" >> /boot/cmdline.txt
   
   # Disable CPU throttling  
   RUN echo "performance" > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   
   # Network tuning
   RUN sysctl -w net.core.busy_poll=50
   RUN sysctl -w net.core.busy_read=50
   ```

**Monitoring & Validation:**
```python
# Latency measurement points
measurement_points = {
    'ue_to_gnb': 'timestamp at PDCP',
    'gnb_to_upf': 'GTP-U header timestamp', 
    'upf_to_mec': 'packet capture at MEC',
    'mec_processing': 'app instrumentation',
    'return_path': 'reverse timestamps'
}
```

**Judging Criteria**:
- [ ] Realistic latency budget breakdown (3 points)
- [ ] Specific radio optimizations for URLLC (2 points)
- [ ] Detailed UPF/transport configuration (2 points)
- [ ] Compute stack optimizations (2 points)
- [ ] Measurement methodology (1 point)

**Red Flags**: Unrealistic latency claims, missing hardware considerations

---

## Regulatory Compliance Domain

### Question 16: Data Privacy (Basic)
**Question**: Explain how to implement GDPR-compliant data retention for CDRs and network logs, including retention periods, anonymization requirements, and audit trails.

**Baseline Answer**:
**GDPR-Compliant CDR & Log Retention Implementation:**

**Retention Periods:**
- CDRs: 6 months (billing purposes)
- Network logs: 3 months (security/troubleshooting)
- Location data: 12 months (anonymized after 4 months)
- Marketing data: Until consent withdrawn

**Implementation Architecture:**
```sql
-- CDR anonymization after 4 months
CREATE TABLE cdr_anonymized AS
SELECT 
    MD5(CONCAT(msisdn, salt)) as hashed_id,
    DATE_TRUNC('hour', call_time) as time_bucket,
    duration,
    cell_id_prefix,  -- Remove last 2 digits
    data_volume
FROM cdr_raw
WHERE call_time < NOW() - INTERVAL '4 months';

-- Auto-deletion policy
CREATE POLICY auto_delete_cdr ON cdr_raw
FOR DELETE TO deletion_service
USING (call_time < NOW() - INTERVAL '6 months');
```

**Audit Trail Requirements:**
1. **Access Logging**
   - Who accessed what data and when
   - Purpose of access (lawful basis)
   - Data exports tracking

2. **Consent Management**
   ```json
   {
     "customer_id": "hash",
     "consents": {
       "marketing": {"status": true, "date": "2024-01-15"},
       "location_analytics": {"status": false, "date": "2024-01-15"}
     },
     "audit": [
       {"action": "consent_given", "type": "marketing", "timestamp": "..."}
     ]
   }
   ```

3. **Right to Erasure Process**
   - Automated deletion workflows
   - Backup synchronization
   - Third-party notification system

**Judging Criteria**:
- [ ] Correct retention periods for different data types (3 points)
- [ ] Anonymization implementation (3 points)
- [ ] Comprehensive audit trails (2 points)
- [ ] Consent management approach (1 point)
- [ ] Automated compliance processes (1 point)

**Red Flags**: Excessive retention periods, no anonymization strategy

---

### Question 17: Lawful Interception (Intermediate)
**Question**: Design LI architecture compliant with ETSI standards, ensuring separation of intercept function, mediation, and law enforcement interfaces. Address both content and metadata.

**Baseline Answer**:
**ETSI-Compliant LI Architecture Design:**

**Three-Function Separation:**

1. **ADMF (Administration Function)**
   - Isolated management network
   - HSM-based warrant storage
   - Role-based access (4-eyes principle)
   - Audit logging to immutable storage

2. **Intercept Functions**
   ```yaml
   # IRI (Intercept Related Information)
   iri_capture:
     sources: [AMF, SMF, UDM]
     data: [IMSI, IMEI, Location, Call_Events]
     format: ETSI_TS_102_232
     
   # CC (Content of Communication)  
   cc_capture:
     sources: [UPF, IMS_Media]
     method: Port_Mirroring
     encryption: Maintained_E2E
   ```

3. **Mediation & Delivery**
   ```
   MDF (Mediation/Delivery Function):
   - Protocol conversion (X1/X2/X3 to HI2/HI3)
   - Encryption to LEA (AES-256 minimum)
   - Reliable delivery (store & forward)
   - Correlation of IRI and CC
   ```

**Security Architecture:**

1. **Network Isolation**
   - Dedicated VRF for LI traffic
   - No routing to production network
   - Encrypted tunnels to LEA

2. **Access Control**
   ```python
   class LIAccessControl:
       def authorize_intercept(self, warrant, target):
           # Verify digital signature
           if not verify_warrant_signature(warrant):
               raise SecurityException("Invalid warrant")
           
           # Check temporal validity
           if not warrant.valid_period.contains(now()):
               raise ExpiredWarrantException()
           
           # Validate target scope
           if target not in warrant.authorized_targets:
               raise ScopeViolationException()
           
           # Log access
           audit_log.record(user, action, warrant_id)
   ```

**Judging Criteria**:
- [ ] Clear three-function separation (3 points)
- [ ] Addresses both IRI and CC (2 points)
- [ ] Security architecture details (2 points)
- [ ] ETSI standard compliance mentions (2 points)
- [ ] Access control implementation (1 point)

**Red Flags**: Mixing functions, no security isolation, missing audit trails

---

### Question 18: Spectrum Compliance (Advanced)
**Question**: Cell site exceeds FCC Part 27 emission limits by 3dB at adjacent channel. Explain remediation options considering: coverage impact, cost, timeline, and regulatory reporting requirements.

**Baseline Answer**:
**FCC Part 27 Emission Violation Remediation:**

**Immediate Actions:**
1. **Reduce power temporarily** (3dB reduction = compliance)
2. **File notification** with FCC within 24 hours
3. **Document everything** for compliance records

**Root Cause Analysis:**
```python
# Emission analysis framework
def analyze_emissions(site_data):
    issues = []
    
    # Check antenna pattern distortion
    if site_data.vswr > 1.5:
        issues.append("Antenna system degradation")
    
    # Verify filter performance
    if site_data.adjacent_channel_power > -45:  # dBc
        issues.append("Filter degradation or bypass")
    
    # Check for PIM
    if site_data.pim_level > -150:  # dBc
        issues.append("PIM from corroded connections")
    
    return issues
```

**Remediation Options (Ranked by Impact/Cost):**

1. **Digital Pre-Distortion Tuning** (1 week, $5K)
   - Recalibrate DPD algorithms
   - Coverage impact: <0.5dB
   - Improves ACLR by 3-5dB

2. **Replace Cavity Filters** (2 weeks, $15K)
   - Install sharper roll-off filters
   - Coverage impact: 1-2dB loss
   - Guarantees 5-10dB improvement

3. **Antenna System Replacement** (4 weeks, $30K)
   - New antenna with better pattern control
   - Coverage impact: Potential improvement
   - Addresses PIM and pattern issues

**Regulatory Reporting:**
```markdown
## FCC Notification Template

Date: [Current]
Licensee: [Company]
Call Sign: [Station]

Non-compliance discovered: [Date]
- Parameter: Adjacent Channel Emissions
- Measured: +3dB over Part 27 limit

Immediate actions:
- Power reduced by 3dB at [time]
- Site now compliant

Permanent resolution:
- [Selected option from above]
- Timeline: [Specific dates]
```

**Judging Criteria**:
- [ ] Immediate compliance actions (3 points)
- [ ] Multiple remediation options with cost/benefit (3 points)
- [ ] Regulatory reporting requirements (2 points)
- [ ] Technical root cause analysis (1 point)
- [ ] Realistic timelines and impacts (1 point)

**Red Flags**: No immediate action, unrealistic fix times, missing FCC notification

---

### Question 19: Emergency Services (Intermediate)
**Question**: Implement E911 Phase 2 location accuracy requirements (50m for 67%, 150m for 95%) in dense urban environment. Address technical solution and compliance verification.

**Baseline Answer**:
**E911 Phase 2 Location Accuracy Implementation:**

**Hybrid Location Solution:**

1. **A-GPS (Assisted GPS)**
   - Primary method for outdoor
   - Accuracy: 5-50m typically
   - Availability: 70% in dense urban

2. **OTDOA (Observed Time Difference of Arrival)**
   - Backup for GPS-denied areas
   - Requires 3+ cells visibility
   - Accuracy: 50-150m

3. **RF Fingerprinting**
   - Database of signal patterns
   - Works indoors/urban canyons
   - Accuracy: 30-100m

4. **Barometric Pressure** (Z-axis)
   - Floor-level accuracy
   - Critical for high-rise buildings

**Implementation Architecture:**
```yaml
location_system:
  smlc:  # Serving Mobile Location Center
    algorithms:
      - agps:
          timeout: 30s
          fallback: otdoa
      - otdoa:
          min_cells: 3
          correlation_window: 5ms
      - rf_fingerprint:
          database_update: daily
          grid_size: 10m
    
  accuracy_enhancement:
    - dense_urban_correction_model
    - multipath_mitigation
    - crowd_sourced_calibration
```

**Compliance Verification Process:**

1. **Test Call Generation**
   ```python
   # Automated testing framework
   test_points = generate_grid(area, spacing=50m)
   
   for point in test_points:
       result = place_test_911_call(point)
       actual_location = point.coordinates
       reported_location = result.location
       error = calculate_distance(actual, reported)
       
       record_result({
           'point': point,
           'error': error,
           'method': result.location_method,
           'confidence': result.confidence
       })
   ```

2. **Statistical Analysis**
   - 67% within 50m (horizontal)
   - 95% within 150m (horizontal)
   - 67% within 3m (vertical)

**Dense Urban Optimizations:**
- Small cell deployment for OTDOA
- Building-specific RF maps
- Coordination with venue owners
- In-building beacon systems

**Judging Criteria**:
- [ ] Multiple location technologies (3 points)
- [ ] Specific accuracy requirements met (2 points)
- [ ] Verification methodology (2 points)
- [ ] Urban-specific optimizations (2 points)
- [ ] Z-axis handling (1 point)

**Red Flags**: Single technology approach, no verification plan

---

### Question 20: Net Neutrality (Advanced)
**Question**: Design QoS framework that maximizes network efficiency while maintaining net neutrality compliance. Address: traffic management, transparency requirements, and technical documentation.

**Baseline Answer**:
**Net Neutrality Compliant QoS Framework:**

**Design Principles:**
1. No blocking of lawful content
2. No throttling based on content/application
3. No paid prioritization
4. Reasonable network management allowed

**Technical Implementation:**

```yaml
# Traffic Management Framework
qos_policies:
  # Class-based, not content-based
  classes:
    network_control:
      dscp: CS6
      priority: strict
      bandwidth: 5%
      use_case: "Network protocols only"
    
    real_time:
      dscp: EF
      priority: high
      bandwidth: 30%
      criteria: "Low latency required (<20ms)"
      examples: "VoIP, gaming, video calls"
    
    streaming:
      dscp: AF41
      priority: medium
      bandwidth: 40%
      criteria: "Sustained throughput"
      examples: "Video streaming, large downloads"
    
    best_effort:
      dscp: 0
      priority: normal
      bandwidth: 25%
      criteria: "Default for all other traffic"

# Application-agnostic classification
classification_rules:
  - match: "packet_size < 100 AND rate > 50pps"
    class: real_time
    reason: "Likely real-time application"
    
  - match: "sustained_rate > 1Mbps AND packet_size > 1000"
    class: streaming
    reason: "Bulk transfer pattern"
```

**Transparency Requirements:**

1. **Public Documentation**
   ```markdown
   ## Network Management Practices
   
   Our network uses class-based QoS that:
   - Prioritizes based on technical requirements
   - Does NOT inspect content/applications
   - Applies equally to all providers
   
   Classes:
   1. Real-time (30%): <20ms latency needs
   2. Streaming (40%): Sustained throughput
   3. Best-effort (25%): Everything else
   ```

2. **Measurement & Reporting**
   ```python
   class NetNeutralityMonitor:
       def verify_compliance(self):
           # Check for discrimination
           for app_type in ['netflix', 'youtube', 'zoom']:
               perf_metrics = measure_performance(app_type)
               if significant_deviation(perf_metrics):
                   alert_compliance_team()
           
           # Generate transparency report
           report = {
               'avg_speeds_by_class': calculate_averages(),
               'congestion_events': get_congestion_stats(),
               'management_actions': get_traffic_shaping_events()
           }
           publish_monthly_report(report)
   ```

**Congestion Management:**
- Apply policies only during actual congestion
- Affect heavy users proportionally
- Exempt real-time traffic from management
- Clear documentation of triggers/thresholds

**Judging Criteria**:
- [ ] Application-agnostic approach (3 points)
- [ ] Clear transparency documentation (2 points)
- [ ] Technical implementation details (2 points)
- [ ] Compliance monitoring approach (2 points)
- [ ] Congestion management strategy (1 point)

**Red Flags**: Content-based discrimination, no transparency measures

---

## Scoring Summary Template

Use this template to score each response:

```
Question #__: [Title]
- Technical Accuracy: __/10
- Completeness: __/10  
- Practical Applicability: __/10
- Domain Expertise: __/10
- Clarity & Structure: __/10

Weighted Score: __/10
Response Time: __s

Strengths:
- 

Weaknesses:
- 

Pass/Fail: [PASS if ≥7.0]
```

## Final Evaluation Framework

**Overall Model Assessment:**
1. **Domain Coverage**: Average score per domain
2. **Complexity Handling**: Score trend (basic→intermediate→advanced)
3. **Telco Competency**: % of responses showing domain expertise
4. **Performance**: Average response time vs. targets
5. **Reliability**: % of coherent, relevant responses

**Recommendation Thresholds:**
- **Production Ready**: Overall ≥8.0, all domains ≥7.5
- **Limited Deployment**: Overall ≥7.0, critical domains ≥7.5  
- **Pilot Only**: Overall ≥6.0, shows improvement potential
- **Not Recommended**: Overall <6.0 or critical failures
