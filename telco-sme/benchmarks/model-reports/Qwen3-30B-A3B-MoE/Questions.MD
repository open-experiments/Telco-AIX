# Expert Telco Questions for Model Validation

## Network Architecture & Standards

**Question 1: 5G Core Architecture**
Explain the difference between Service Based Architecture (SBA) and Control and User Plane Separation (CUPS) in 5G core networks. How does AMF handle registration procedures compared to MME in 4G?

**Question 2: Network Slicing**
Design a network slicing strategy for a carrier supporting eMBB, URLLC, and mMTC services. Include specific QoS parameters, isolation mechanisms, and resource allocation algorithms.

**Question 3: RAN Disaggregation**
Compare O-RAN vs traditional RAN architectures. What are the implications of RIC (RAN Intelligent Controller) deployment for vendor lock-in and CAPEX/OPEX optimization?

## RF Engineering & Optimization

**Question 4: Massive MIMO**
Calculate the theoretical capacity improvement for a 64T64R massive MIMO system vs 4T4R in a dense urban environment. Consider beamforming gains, interference mitigation, and spectral efficiency.

**Question 5: Cell Planning**
You're deploying mmWave 5G in Manhattan. Design the cell planning strategy considering building penetration loss, atmospheric attenuation, and mobility handover challenges.

**Question 6: Interference Management**
Explain Inter-Cell Interference Coordination (ICIC) vs enhanced ICIC (eICIC) vs Further enhanced ICIC (FeICIC). When would you deploy each technique?

## Protocol Deep Dive

**Question 7: Signaling Optimization**
Analyze the signaling overhead differences between 4G attach procedures vs 5G registration. How does the AMF's connection management compare to MME's context management?

**Question 8: Transport Networks**
Design a transport network architecture for a 5G deployment using segment routing with IPv6 (SRv6). Include traffic engineering, low-latency requirements, and failure recovery mechanisms.

**Question 9: Edge Computing Integration**
How would you integrate Multi-Access Edge Computing (MEC) with a 5G SA core? Address service continuity during UE mobility between MEC sites.

## Business & Operations

**Question 10: Spectrum Strategy**
A carrier has 100MHz in C-band (3.5GHz), 200MHz in mmWave (28GHz), and legacy 2G/3G/4G spectrum. Develop a spectrum refarming and deployment strategy for maximum revenue per MHz.

**Question 11: Network Economics**
Calculate the business case for Open RAN vs traditional RAN for a greenfield deployment covering 10,000 cell sites. Include TCO analysis over 7 years.

**Question 12: Service Assurance**
Design a machine learning-based network optimization system that can predict and prevent service degradation. What KPIs would you monitor and what actions would you automate?

## Advanced Technical Topics

**Question 13: Voice Architecture**
Compare VoLTE vs VoNR implementation strategies. How does 5G voice call continuity work during inter-RAT handovers to 4G networks without VoLTE?

**Question 14: Private Networks**
Design a private 5G network for a manufacturing facility requiring ultra-low latency (sub-1ms) for robotic control. Include spectrum options, core network placement, and security architecture.

**Question 15: Cloud-Native Transformation**
Architect the migration of a legacy telco infrastructure to cloud-native 5G. Address decomposition of monolithic network functions, microservices design, and CI/CD pipelines for network functions.

## Evaluation Criteria

**Technical Accuracy**: Correct use of 3GPP terminology, standards compliance, accurate technical specifications

**Depth of Knowledge**: Understanding of protocol internals, RF engineering principles, network optimization techniques

**Practical Application**: Real-world feasibility, cost considerations, operational complexity

**Standards Alignment**: 3GPP Release 15/16/17 compliance, O-RAN specifications, ETSI standards

**Innovation Awareness**: Knowledge of latest developments, emerging technologies, future roadmaps
