# Diffusion Model Recommendation System

A comprehensive, research-validated system for selecting optimal diffusion models for influence maximization on temporal networks.

## 📚 Based on Research

This system implements recommendations from:
**"Diffusion Models for Influence Maximization on Temporal Networks: A Guide to Make the Best Choice"**

The system includes:
- **86 diffusion models** from the paper's taxonomy
- **Empirical validation** from 9 real-world datasets
- **3 validated scenarios**: Rumor spreading, Disease progression, Viral marketing

---

## 🚀 Quick Start

```bash
python diffusion_model_recommender.py
```

Follow the interactive questionnaire to receive personalized model recommendations.

---

## 📊 Key Features

### 1. Comprehensive Model Database
- **Process-Oriented Models** (21 models)
  - Explanatory: SI, SIR, SEIR, SCIR, cpSI-R, etc.
  - Cascade: IC, CT-IC, TCC, ASIC, CTM-IC
  - Threshold: LT, MTM, STM, DLT, pELT, tELT, CLT, DRUC
  
- **Interaction-Oriented Models** (12 models)
  - Pairwise: Voter, EVM, DVM, BDVM, IEM, OM-WTD, LIM, POE
  - Group: TAM, TLRA, PCL-DC, CTMM
  
- **Competition-Oriented Models** (10 models)
  - DBM, WPM, WPTM, DCM, TIC, AtI, TrCID, TICC, TBCELF, IML-IC
  
- **Structure-Oriented Models** (9 models)
  - Micro: AgentBased, LND, LowClustering, PAM
  - Macro: HighClustering, DensityBased, BIADM, ABBM, PAAM
  
- **Target-Oriented Models** (14 models)
  - VMID, MAT, UAD, FSC-SB, FSC-N, FST-SB, FST-N, IC-u, LT-u, CAND, ISR, ACT

### 2. Scenario-Specific Recommendations

Pre-configured recommendations for 14 scenarios:
- Rumor spreading
- Disease progression
- Viral marketing
- Competitive marketing
- Antimicrobial resistance
- Cancer progression
- Opinion dynamics
- Influencer marketing
- Misinformation control
- Emergency alerts
- Political campaigns
- Product adoption
- Budget-constrained scenarios

### 3. Research-Validated Performance

Based on empirical analysis from the paper:

**Higgs Twitter (Rumor Spreading)**
- Best: irSIR, UAD, SI, SIR, EVM
- Worst: ABBM, SCM
- Key insight: Interaction-aware models best reproduce social rumor dynamics

**Hospital Ward (Disease Progression)**
- Best: SCM, LT, SI, WPM, TAM
- Worst: SIR, UAD, POE
- Key insight: Structurally driven models excel in deterministic epidemic settings

**CollegeMsg (Viral Marketing)**
- Best: SCM, LT, SI, AtI, MAT
- Worst: irSIR, PAM, IC
- Key insight: Target-oriented models most suitable for viral marketing

### 4. Multi-Factor Scoring

Recommendations based on:
- Scenario match (5.0 points)
- Temporal support (2.5 points)
- Competition handling (3.5 points)
- Behavioral modeling (3.0 points)
- Goal alignment (up to 4.0 points)
- Paper validation (3.0 points)
- Network size compatibility (2.0 points)
- Special features (up to 3.5 points)

### 5. Adaptive Learning

System learns from user feedback:
- Performance ratings
- Aspect-specific feedback
- Issue identification
- Weight adjustments (-0.5 to +0.8)

---

## 📖 Usage Guide

### Interactive Mode

Run the program and answer 13 questions:

1. **Primary Scenario** - Select from 12 pre-defined scenarios
2. **Network Type** - Temporal, static, or dynamic
3. **Ground Truth** - Real data, idealized, or none
4. **Primary Goal** - Accuracy, reach, fairness, efficiency, cost, or early accuracy
5. **Competition** - Yes/No
6. **Behavioral Factors** - Yes/No
7. **Community Structure** - Yes/No
8. **Seed Constraints** - None, budget, fairness, or targeted
9. **Temporal Priority** - Early, final, peak, or full trajectory
10. **Approximation Guarantees** - Required? Yes/No
11. **Network Size** - Small, medium, large, or very large
12. **Special Features** - Reinfection, recovery, latency, trust, memory, or none
13. **Learning** - Enable feedback learning? Yes/No

### Example Session

```
DIFFUSION MODEL RECOMMENDATION SYSTEM

1) What is your PRIMARY application scenario?
   a) Rumor/information spreading on social media
   ...
   Enter choice (a-l): a

2) What type of network are you working with?
   a) Temporal network (time-stamped interactions)
   ...
   Enter choice (a-c): a

3) Do you have real diffusion ground truth data?
   a) Yes, empirical diffusion traces
   ...
   Enter choice (a-c): a

...

TOP RECOMMENDED MODELS
======================================================================

Scenario: Rumor Spreading
Expert Insight: Paper shows irSIR achieves lowest RMSE on Higgs 
Twitter dataset. EVM performs well for community-driven rumor dynamics.

1. irSIR
   Score: 12.50
   Category: Process-Explanatory
   Description: Infection rate depends on recovered population - 
                BEST for rumor spreading
   Why recommended: Top recommendation for rumor_spreading; 
                     Supports temporal dynamics; Incorporates 
                     behavioral dynamics; ★ Validated: Top performer 
                     on Higgs Twitter
   Properties: Temporal, Behavioral, Guarantees ✓
   Complexity: Polynomial
   Coverage: Medium
   Temporal Accuracy: Very High
```

---

## 🎯 Validation Scenarios

### Scenario 1: Social Media Rumor Spreading

**Setup:**
- Dataset: Higgs Twitter (456,626 nodes, 14,855,842 edges)
- Ground truth: Real diffusion traces
- Task: Model information cascade during Higgs boson discovery

**Top Performers (k=20):**
1. UAD (RMSE: 0.0921)
2. SI (RMSE: 0.1267)
3. MTM (RMSE: 0.2221)
4. SIR (RMSE: 0.1503)
5. LND (RMSE: 0.2821)

**Key Findings:**
- irSIR achieves lowest overall RMSE across seed sizes
- Interaction-aware models (EVM, TAM, POE) perform well
- Structure-oriented models (ABBM) fail to capture social dynamics
- Temporal accuracy matters more than final coverage

**Recommended Models:**
- Primary: irSIR, EVM
- Alternative: UAD, TAM, POE

---

### Scenario 2: Disease Spread & Intervention

**Setup:**
- Datasets: Hospital Ward (75 nodes), SFHH Conference (405 nodes)
- Ground truth: IC(p=1) as worst-case scenario
- Task: Model face-to-face disease transmission

**Top Performers (Hospital Ward, k=20):**
1. TAM (RMSE: 0.0200)
2. SI (RMSE: 0.0200)
3. WPM (RMSE: 0.0200)
4. SEIR (RMSE: 0.0208)
5. SCM (RMSE: 0.0310)

**Key Findings:**
- SCM and LT achieve near-zero error
- Structure-driven models dominate
- Recovery-based models (SIR) introduce unnecessary delay
- Deterministic propagation best for planning

**Recommended Models:**
- Primary: SCM, LT, SI
- For latency: SEIR
- For reinfection: cpSI-R
- For budget constraints: TBCELF

---

### Scenario 3: Viral Marketing Campaign

**Setup:**
- Dataset: CollegeMsg (1,899 nodes, 59,835 edges)
- Ground truth: IC(p=1)
- Task: Maximize product adoption in university network

**Top Performers (k=20):**
1. SCM (RMSE: 0.0200)
2. LT (RMSE: 0.0200)
3. SI (RMSE: 0.0227)
4. AtI (RMSE: 0.0469)
5. SEIR (RMSE: 0.0261)

**Key Findings:**
- SCM consistently most accurate
- Target-oriented models excel (VMID, MAT, UAD)
- irSIR performs poorly (mismatch with broadcast-style spread)
- Coverage-oriented metrics don't guarantee temporal accuracy

**Recommended Models:**
- Primary: VMID, SCM, MAT
- For user engagement: IC-u, LT-u, UAD
- For competition: TBCELF, TIC, AtI

---

## 📈 Model Selection Decision Tree

```
START
  │
  ├─ Rumor Spreading?
  │   ├─ Yes → irSIR, EVM, TAM, POE
  │   └─ No ↓
  │
  ├─ Disease Modeling?
  │   ├─ Yes → SCM, LT, SI, SEIR
  │   └─ No ↓
  │
  ├─ Marketing Campaign?
  │   ├─ Single brand → VMID, SCM, MAT, UAD
  │   └─ Multiple competitors → TBCELF, TIC, AtI
  │
  ├─ Budget Constrained?
  │   ├─ Yes → TBCELF (★ BEST)
  │   └─ No ↓
  │
  ├─ Need Guarantees?
  │   ├─ Yes → Filter by submodular + monotone
  │   └─ No ↓
  │
  ├─ Temporal Network?
  │   ├─ Yes → pELT, CT-IC, TCC, cpSI-R
  │   └─ Static → IC, LT, SIR
  │
  ├─ Behavioral Factors?
  │   ├─ Yes → Voter, EVM, DVM, POE, LIM
  │   └─ No → IC, SI, LT
  │
  └─ Large Network (>100k)?
      ├─ Yes → ASIC, SI, Polynomial models
      └─ No → Any model
```

---

## 🔧 Advanced Features

### 1. Export Recommendations

After receiving recommendations:
```
Would you like to export these recommendations? (yes/no): yes
✓ Recommendations exported to: recommendations_rumor_spreading.json
```

**Export Format:**
```json
{
  "scenario": "rumor_spreading",
  "timestamp": "2026-01-29T...",
  "user_answers": {...},
  "recommendations": [
    {
      "rank": 1,
      "model": "irSIR",
      "score": 12.5,
      "reasoning": "Top recommendation for rumor_spreading; ...",
      "properties": {...}
    }
  ]
}
```

### 2. Feedback Learning

The system adapts based on your feedback:

```
FEEDBACK
======================================================================

You selected: irSIR
Please provide feedback to improve future recommendations:

1) How well did this model perform?
   a) Excellent - exceeded expectations
   b) Good - met expectations
   c) Fair - acceptable but could be better
   d) Poor - did not meet needs
   Enter choice (a-d): a

2) What worked particularly well? (select multiple: e.g., a,c)
   a) Temporal accuracy
   b) Coverage/reach
   c) Computational efficiency
   d) Ease of implementation
   e) Model interpretability
   Enter choices: a,b

✓ Feedback recorded. The system will provide better recommendations.
```

**Weight Updates:**
- Excellent → +0.8
- Good → +0.3
- Fair → -0.2
- Poor → -0.5

### 3. Model Properties Database

Each model includes:
- **Taxonomy**: Classification category
- **Temporal support**: Yes/No
- **Competitive**: Handles multiple campaigns
- **Behavioral**: Incorporates user behavior
- **Submodular/Monotone**: Guarantees available
- **Complexity**: Polynomial or NP-hard
- **Best for**: Specific use cases
- **Coverage**: Expected reach (low/medium/high/very_high)
- **Temporal accuracy**: Time-series fidelity (low/medium/high/very_high)
- **Description**: Model explanation

---

## 📊 Performance Metrics

### From Paper Experiments

**Metric Definitions:**
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **Peak Error**: Deviation at maximum diffusion (lower is better)
- **Final Error**: Deviation at saturation (lower is better)
- **FAF**: Final Activated Fraction (higher is better)

**Best Models by Metric:**

| Metric | Rumor (Higgs) | Disease (Hospital) | Marketing (College) |
|--------|---------------|-------------------|---------------------|
| RMSE | irSIR (0.0851) | TAM (0.0200) | SCM (0.0200) |
| MAE | irSIR (0.0635) | TAM (0.0200) | SCM (0.0200) |
| Peak | irSIR (0.1241) | Multiple (0.0200) | Multiple (0.0200) |
| Final | irSIR (0.1241) | Multiple (0.0200) | Multiple (0.0200) |
| FAF | MTM (0.9165) | Multiple (0.9800) | Multiple (0.9800+) |

---

## 🔬 Research Insights

### Key Findings from the Paper

1. **No universal model**: Model suitability is scenario-dependent

2. **Rumor spreading**: Interaction-aware models (irSIR, EVM) achieve superior temporal fidelity

3. **Disease progression**: Structure-driven models (SCM, LT) excel under deterministic spread

4. **Viral marketing**: Target-oriented models (VMID, SCM) provide best accuracy

5. **Diminishing returns**: Accuracy gains diminish beyond k ≈ 15 seed nodes

6. **Coverage ≠ Accuracy**: High final activation doesn't guarantee temporal correctness

7. **Early advantage**: Models capturing initial spread patterns perform best overall

### Model Taxonomy Insights

**Process-Oriented**
- Fast saturation in early snapshots
- Good for time-critical scenarios
- MTM shows delayed cascades in trust networks

**Interaction-Oriented**
- Steady convergence toward saturation
- Step-wise growth reflects discrete interactions
- Sensitive to interaction semantics

**Competition-Oriented**
- Aggressive early acquisition (60-70% in 2-3 snapshots)
- Tight convergence despite algorithmic diversity
- High sensitivity to seed set expansion

**Structure-Oriented**
- Delayed acceleration with staircase patterns
- Community-driven saturation plateaus
- Excellent in low-clustering networks

**Target-Oriented**
- Near-optimal saturation
- Rapid early adoption (60% in 2 snapshots)
- Tight clustering across specialized objectives

---

## 🛠️ Customization

### Adding Custom Scenarios

Edit `SCENARIO_RECOMMENDATIONS` in the code:

```python
SCENARIO_RECOMMENDATIONS = {
    "my_custom_scenario": {
        "top_models": ["Model1", "Model2", "Model3"],
        "reasoning": "Why these models work best...",
        "avoid": ["BadModel1", "BadModel2"],
        "metrics_priority": ["accuracy", "coverage"]
    }
}
```

### Adding New Models

Add to `MODELS` dictionary:

```python
"MyModel": {
    "taxonomy": "Process-Explanatory",
    "temporal": True,
    "competitive": False,
    "behavioral": True,
    "submodular": True,
    "monotone": True,
    "complexity": "NP-hard",
    "best_for": ["my_use_case"],
    "coverage": "high",
    "temporal_accuracy": "very_high",
    "description": "My model description"
}
```

---

## 📝 Limitations

1. **Model availability**: You must implement the recommended models yourself
2. **Parameter tuning**: Models may require dataset-specific parameter optimization
3. **Computational resources**: Some models (NP-hard) may be slow on large networks
4. **Ground truth**: Results depend on availability of validation data
5. **Static recommendations**: Assumes network properties remain stable

---

## 🔮 Future Enhancements

Potential additions:
- [ ] Automatic parameter tuning
- [ ] Model ensemble recommendations
- [ ] Multi-objective optimization
- [ ] Real-time performance monitoring
- [ ] Integration with network analysis tools
- [ ] Visualization of diffusion dynamics
- [ ] Cross-validation support
- [ ] Hybrid model suggestions

---

## 📚 References

**Primary Paper:**
"Diffusion Models for Influence Maximization on Temporal Networks: A Guide to Make the Best Choice"
Authors: Aaqib Zahoor, Iqra Altaf Gillani, Janibul Bashir
Institution: National Institute of Technology, Srinagar

**Key Citations:**
- Kempe et al. (2003): Independent Cascade & Linear Threshold
- Holley & Liggett (1975): Voter Model
- Bass (1969): Bass Diffusion Model
- Granovetter (1978): Threshold Models

---

## 💡 Tips for Best Results

1. **Be specific**: Choose the most relevant scenario
2. **Consider trade-offs**: High accuracy may mean lower efficiency
3. **Validate**: Test recommendations on your specific dataset
4. **Iterate**: Use feedback to improve future recommendations
5. **Combine insights**: Consider top 3-5 models, not just #1
6. **Check paper**: Refer to original paper for implementation details
7. **Start simple**: Begin with polynomial-complexity models for large networks
8. **Match goals**: Align model selection with your primary objective

---

## ❓ FAQ

**Q: Why isn't my preferred model ranked #1?**
A: The system uses multi-factor scoring based on your scenario. Your model may not match all specified criteria.

**Q: How often should I update weights?**
A: Provide feedback after each use for best learning.

**Q: Can I use this for networks with >1M nodes?**
A: Yes, but prioritize polynomial-complexity models (SI, SIR, SEIR, Voter, EVM).

**Q: What if I don't have ground truth data?**
A: Select "idealized" (IC with p=1) or "none" - the system will still provide valid recommendations.

**Q: Are these models implemented in the code?**
A: Yes, this is a recommendation system. You can implement the models by selecting the datasets and python files for choosen class.

**Q: How accurate are these recommendations?**
A: Based on empirical validation across 9 real-world datasets with 86 models. See paper for details.

---

## 📧 Support

For questions about:
- **The recommendation system**: Check this README first
- **Model implementation**: Refer to the original paper and cited references
- **Research insights**: Consult the paper's experimental sections
- **Custom scenarios**: See customization section above

---

## 📄 License

This recommendation system is provided as-is for research and educational purposes.
Refer to the original paper for academic citation requirements.

---

## 🙏 Acknowledgments

Based on comprehensive research by Aaqib Zahoor, Iqra Altaf Gillani, and Janibul Bashir at the National Institute of Technology, Srinagar.

Validated using datasets from:
- Stanford SNAP (Twitter, Email, Bitcoin networks)
- SocioPatterns (Hospital, Conference, Malawi networks)
- MOOC User Action dataset

---

**Version:** 1.0  
**Last Updated:** January 2026  
**Python Version:** 3.7+  
**Dependencies:** None (pure Python)
