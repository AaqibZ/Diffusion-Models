import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any

# ============================
# COMPREHENSIVE MODEL KNOWLEDGE BASE
# ============================

MODELS = {
    # Process-Oriented: Explanatory Models
    "SI": {
        "taxonomy": "Process-Explanatory",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["epidemic_simple", "irreversible_spread"],
        "coverage": "high",
        "temporal_accuracy": "medium",
        "description": "Susceptible-Infected model for irreversible contagion"
    },
    "SIR": {
        "taxonomy": "Process-Explanatory",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["disease_with_recovery", "realistic_epidemics"],
        "coverage": "high",
        "temporal_accuracy": "medium",
        "description": "Captures immunity effects and recovery dynamics"
    },
    "SIS": {
        "taxonomy": "Process-Explanatory",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["persistent_contagion", "recurrent_disease"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Models cyclic infection without permanent immunity"
    },
    "SEIR": {
        "taxonomy": "Process-Explanatory",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["disease_with_latency", "incubation_delay"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Includes exposed (latent) state before infection"
    },
    "SCIR": {
        "taxonomy": "Process-Explanatory",
        "temporal": False,
        "competitive": True,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["antimicrobial_resistance", "competitive_spread"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Competitive infection and recovery dynamics"
    },
    "irSIR": {
        "taxonomy": "Process-Explanatory",
        "temporal": False,
        "competitive": False,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["feedback_driven", "saturation_capture"],
        "coverage": "medium",
        "temporal_accuracy": "very_high",
        "description": "Infection rate depends on recovered population - BEST for rumor spreading"
    },
    "FSIR": {
        "taxonomy": "Process-Explanatory",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["hub_control", "degree_normalized"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Prevents hub dominance through degree normalization"
    },
    "cpSI-R": {
        "taxonomy": "Process-Explanatory",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["reinfection", "partial_immunity"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Temporal SI with partial recovery and reactivation"
    },
    "ESIS": {
        "taxonomy": "Process-Explanatory",
        "temporal": False,
        "competitive": False,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["external_influence", "exogenous_effects"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Epidemic SIS with external stimuli"
    },
    
    # Process-Oriented: Predictive Cascade Models
    "IC": {
        "taxonomy": "Process-Predictive-Cascade",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["viral_marketing", "baseline_cascade"],
        "coverage": "very_high",
        "temporal_accuracy": "medium",
        "description": "Baseline probabilistic cascade model"
    },
    "CT-IC": {
        "taxonomy": "Process-Predictive-Cascade",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["time_windows", "continuous_time"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Continuous-time IC with activation windows"
    },
    "TCC": {
        "taxonomy": "Process-Predictive-Cascade",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["memory_effects", "reinforcement"],
        "coverage": "high",
        "temporal_accuracy": "very_high",
        "description": "Time-Dependent Comprehensive Cascade with memory"
    },
    "ASIC": {
        "taxonomy": "Process-Predictive-Cascade",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["scalability", "large_networks"],
        "coverage": "high",
        "temporal_accuracy": "medium",
        "description": "Scalable IC approximation for large networks"
    },
    "CTM-IC": {
        "taxonomy": "Process-Predictive-Cascade",
        "temporal": True,
        "competitive": True,
        "behavioral": False,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["multi_channel", "continuous_diffusion"],
        "coverage": "high",
        "temporal_accuracy": "very_high",
        "description": "Continuous-time Markov chain IC"
    },
    
    # Process-Oriented: Predictive Threshold Models
    "LT": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["collective_behavior", "threshold_adoption"],
        "coverage": "very_high",
        "temporal_accuracy": "medium",
        "description": "Canonical threshold model for adoption"
    },
    "MTM": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["voting", "consensus"],
        "coverage": "very_high",
        "temporal_accuracy": "low",
        "description": "Majority Threshold Model for consensus systems"
    },
    "STM": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["fast_triggering", "rapid_spread"],
        "coverage": "very_high",
        "temporal_accuracy": "low",
        "description": "Small Threshold Model for rapid cascades"
    },
    "UTM": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["containment", "conservative_diffusion"],
        "coverage": "low",
        "temporal_accuracy": "high",
        "description": "Unanimous Threshold for strict activation"
    },
    "DLT": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["misinformation", "dynamic_thresholds"],
        "coverage": "medium",
        "temporal_accuracy": "high",
        "description": "Dynamic Linear Threshold with evolving thresholds"
    },
    "pELT": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["persistent_influence", "accumulation"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Persistent Evolving Linear Threshold - accumulates influence"
    },
    "tELT": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["burst_events", "snapshot_based"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Transient Evolving Linear Threshold - no accumulation"
    },
    "CLT": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": False,
        "competitive": True,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["adversarial", "competitive_markets"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Competitive Linear Threshold"
    },
    "DRUC": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": False,
        "competitive": False,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["content_aware", "user_centric"],
        "coverage": "very_high",
        "temporal_accuracy": "high",
        "description": "Decaying Reinforced User-Centric model"
    },
    "LTC": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": False,
        "competitive": False,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["experience_based", "context_aware"],
        "coverage": "high",
        "temporal_accuracy": "medium",
        "description": "Linear Threshold with Color (user experience)"
    },
    "OCM": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["heuristic_selection", "efficiency"],
        "coverage": "high",
        "temporal_accuracy": "medium",
        "description": "Potential Influence Nodes heuristic"
    },
    "GTM": {
        "taxonomy": "Process-Predictive-Threshold",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["diverse_cascades", "markov_transitions"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Generalized Threshold via Markov chains"
    },
    "SCM": {
        "taxonomy": "Process-Explanatory",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["higher_order", "group_interactions"],
        "coverage": "very_high",
        "temporal_accuracy": "very_high",
        "description": "Simplicial Complex Model - BEST for disease progression"
    },
    
    # Interaction-Oriented: Pairwise Models
    "Voter": {
        "taxonomy": "Interaction-Pairwise",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["consensus", "opinion_formation"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Canonical consensus model via neighbor imitation"
    },
    "EVM": {
        "taxonomy": "Interaction-Pairwise",
        "temporal": False,
        "competitive": False,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["community_aware", "modular_networks"],
        "coverage": "high",
        "temporal_accuracy": "very_high",
        "description": "Extended Voter Model for clustered networks - BEST for rumor (after irSIR)"
    },
    "DVM": {
        "taxonomy": "Interaction-Pairwise",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "Polynomial",
        "best_for": ["dynamic_networks", "rewiring"],
        "coverage": "medium",
        "temporal_accuracy": "high",
        "description": "Dynamic Voter Model with network evolution"
    },
    "BDVM": {
        "taxonomy": "Interaction-Pairwise",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "Polynomial",
        "best_for": ["biased_consensus", "majority_preference"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Biased Dynamic Voter Model"
    },
    "IEM": {
        "taxonomy": "Interaction-Pairwise",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["strategic_decisions", "learning_tradeoff"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Information Exchange Model with delayed actions"
    },
    "OM-WTD": {
        "taxonomy": "Interaction-Pairwise",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "Polynomial",
        "best_for": ["non_markovian", "heavy_tailed"],
        "coverage": "medium",
        "temporal_accuracy": "high",
        "description": "Opinion Model with arbitrary Waiting-Time Distributions"
    },
    "LIM": {
        "taxonomy": "Interaction-Pairwise",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["role_based", "leader_identification"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Leader Influence Maximization"
    },
    "TLRA": {
        "taxonomy": "Interaction-Group",
        "temporal": False,
        "competitive": False,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["topic_specific", "content_influence"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Topic-Leader Rank Algorithm"
    },
    
    # Interaction-Oriented: Group Models
    "TAM": {
        "taxonomy": "Interaction-Group",
        "temporal": False,
        "competitive": False,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["hashtag_diffusion", "topic_adoption"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Topic Adoption Model via random walks"
    },
    "POE": {
        "taxonomy": "Interaction-Pairwise",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "Polynomial",
        "best_for": ["preference_driven", "peer_influence"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Preference-Oriented Exposure model"
    },
    "PCL-DC": {
        "taxonomy": "Interaction-Group",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["community_detection", "hybrid_clustering"],
        "coverage": "medium",
        "temporal_accuracy": "low",
        "description": "Probabilistic Content-Link clustering"
    },
    "CTMM": {
        "taxonomy": "Interaction-Pairwise",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "Polynomial",
        "best_for": ["mobility_aware", "temporal_communities"],
        "coverage": "medium",
        "temporal_accuracy": "high",
        "description": "Continuous-Time Markov Model with mobility"
    },
    
    # Competition-Oriented Models
    "DBM": {
        "taxonomy": "Competition",
        "temporal": False,
        "competitive": True,
        "behavioral": False,
        "submodular": True,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["facility_location", "distance_based"],
        "coverage": "medium",
        "temporal_accuracy": "low",
        "description": "Distance-Based competitive model"
    },
    "WPM": {
        "taxonomy": "Competition",
        "temporal": False,
        "competitive": True,
        "behavioral": False,
        "submodular": True,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["wave_propagation", "spatial_spread"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Wave Propagation Model"
    },
    "WPTM": {
        "taxonomy": "Competition",
        "temporal": False,
        "competitive": True,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["heterogeneous_competition", "weighted_thresholds"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Weight-Proportional Threshold Model"
    },
    "DCM": {
        "taxonomy": "Competition",
        "temporal": False,
        "competitive": True,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["decision_aware", "thinking_state"],
        "coverage": "medium",
        "temporal_accuracy": "high",
        "description": "Decision-aware Competitive Model with deliberation"
    },
    "TIC": {
        "taxonomy": "Competition",
        "temporal": True,
        "competitive": True,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["fair_competition", "multi_party"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Timeliness Independent Cascade for fair influence"
    },
    "AtI": {
        "taxonomy": "Competition",
        "temporal": False,
        "competitive": True,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["awareness_phase", "two_stage"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Awareness-to-Influence model"
    },
    "TrCID": {
        "taxonomy": "Competition",
        "temporal": True,
        "competitive": True,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["trust_distrust", "misinformation"],
        "coverage": "medium",
        "temporal_accuracy": "high",
        "description": "Trust-aware Competitive Influence Diffusion"
    },
    "TICC": {
        "taxonomy": "Competition",
        "temporal": True,
        "competitive": True,
        "behavioral": False,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["global_competition", "simplified_modeling"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Targeted Influence Competition Cascade"
    },
    "TBCELF": {
        "taxonomy": "Competition",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["budget_constrained", "cost_effective"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Temporal Budget-aware Cost-Effective Lazy Forward - BEST for targeted intervention"
    },
    "IML-IC": {
        "taxonomy": "Competition",
        "temporal": False,
        "competitive": True,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["unwanted_users", "efficient_estimation"],
        "coverage": "high",
        "temporal_accuracy": "medium",
        "description": "Influence Maximization with Limited unwanted users"
    },
    
    # Structure-Oriented: Micro Models
    "AgentBased": {
        "taxonomy": "Structure-Micro",
        "temporal": False,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["utility_driven", "rational_behavior"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Agent-Based Utility Model"
    },
    "LND": {
        "taxonomy": "Structure-Micro",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["local_neighborhoods", "peer_influence"],
        "coverage": "medium",
        "temporal_accuracy": "high",
        "description": "Local Neighborhood Diffusion"
    },
    "LowClustering": {
        "taxonomy": "Structure-Micro",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["sparse_networks", "early_stage"],
        "coverage": "low",
        "temporal_accuracy": "medium",
        "description": "Low Clustering Diffusion Model"
    },
    "PAM": {
        "taxonomy": "Structure-Micro",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["product_adoption", "agent_heterogeneity"],
        "coverage": "medium",
        "temporal_accuracy": "high",
        "description": "Product Adopter Model"
    },
    
    # Structure-Oriented: Macro Models
    "HighClustering": {
        "taxonomy": "Structure-Macro",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["dense_networks", "coexistence"],
        "coverage": "medium",
        "temporal_accuracy": "low",
        "description": "High Clustering Model for multiple equilibria"
    },
    "DensityBased": {
        "taxonomy": "Structure-Macro",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["mass_adoption", "global_feedback"],
        "coverage": "high",
        "temporal_accuracy": "low",
        "description": "Density-Based Diffusion Model"
    },
    "BIADM": {
        "taxonomy": "Structure-Macro",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["aggregate_adoption", "market_forecasting"],
        "coverage": "high",
        "temporal_accuracy": "low",
        "description": "Bass Innovation-Adoption Diffusion Model"
    },
    "ABBM": {
        "taxonomy": "Structure-Macro",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "Polynomial",
        "best_for": ["network_bass", "micro_macro"],
        "coverage": "high",
        "temporal_accuracy": "medium",
        "description": "Agent-Based Bass Model"
    },
    "PAAM": {
        "taxonomy": "Structure-Macro",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": True,
        "monotone": True,
        "complexity": "NP-hard",
        "best_for": ["bridge_targeting", "structural_exploration"],
        "coverage": "high",
        "temporal_accuracy": "medium",
        "description": "Product Agent Adoption Model"
    },
    
    # Target-Oriented Models
    "VMID": {
        "taxonomy": "Target",
        "temporal": False,
        "competitive": False,
        "behavioral": False,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["viral_marketing", "business_oriented"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Viral Marketing Information Diffusion - BEST for marketing"
    },
    "MAT": {
        "taxonomy": "Target",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["time_aware_marketing", "seasonal_campaigns"],
        "coverage": "high",
        "temporal_accuracy": "high",
        "description": "Multi-agent Trust-based targeting"
    },
    "UAD": {
        "taxonomy": "Target",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["awareness_tendency", "two_stage_adoption"],
        "coverage": "high",
        "temporal_accuracy": "very_high",
        "description": "User-Aware Diffusion with awareness-tendency stages"
    },
    "FSC-SB": {
        "taxonomy": "Target",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["sentiment_control", "blocking"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Fuzzy Sign-aware Cascade with blocking"
    },
    "FSC-N": {
        "taxonomy": "Target",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["negative_influence", "polarity"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Fuzzy Sign-aware Cascade with negative users"
    },
    "FST-SB": {
        "taxonomy": "Target",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["trust_suppression", "fuzzy_thresholds"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Fuzzy Sign-aware Threshold with blocking"
    },
    "FST-N": {
        "taxonomy": "Target",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["adversarial_spread", "temporal_polarity"],
        "coverage": "medium",
        "temporal_accuracy": "high",
        "description": "Fuzzy Sign-aware Threshold with negative activation"
    },
    "IC-u": {
        "taxonomy": "Target",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["engagement_aware", "personalized_IC"],
        "coverage": "high",
        "temporal_accuracy": "medium",
        "description": "Independent Cascade with user engagement"
    },
    "LT-u": {
        "taxonomy": "Target",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["user_thresholds", "personalized_LT"],
        "coverage": "high",
        "temporal_accuracy": "medium",
        "description": "Linear Threshold with user-specific weights"
    },
    "CAND": {
        "taxonomy": "Target",
        "temporal": True,
        "competitive": False,
        "behavioral": False,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["cellular_automaton", "spatial_diffusion"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Cellular Automaton Network Diffusion"
    },
    "ISR": {
        "taxonomy": "Target",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["opinion_evolution", "political_analysis"],
        "coverage": "medium",
        "temporal_accuracy": "medium",
        "description": "Information Spread Reinforcement"
    },
    "ACT": {
        "taxonomy": "Target",
        "temporal": True,
        "competitive": False,
        "behavioral": True,
        "submodular": False,
        "monotone": False,
        "complexity": "NP-hard",
        "best_for": ["learning_based", "rl_influence"],
        "coverage": "medium",
        "temporal_accuracy": "high",
        "description": "Actor-Critic Trust diffusion with RL"
    },
}

# ============================
# SCENARIO-SPECIFIC RECOMMENDATIONS
# ============================

SCENARIO_RECOMMENDATIONS = {
    "rumor_spreading": {
        "top_models": ["irSIR", "EVM", "TAM", "POE"],
        "reasoning": "Paper shows irSIR achieves lowest RMSE on Higgs Twitter dataset. EVM performs well for community-driven rumor dynamics.",
        "avoid": ["MTM", "ABBM"],
        "metrics_priority": ["temporal_accuracy", "coverage"]
    },
    "disease_progression": {
        "top_models": ["SCM", "LT", "SI", "SEIR", "WPM", "TAM"],
        "reasoning": "Paper shows SCM and LT achieve near-zero error under deterministic IC(p=1). Structure-driven models excel in disease scenarios.",
        "avoid": ["irSIR", "PAM"],
        "metrics_priority": ["temporal_accuracy", "coverage"]
    },
    "viral_marketing": {
        "top_models": ["VMID", "SCM", "MAT", "UAD", "IC", "LT"],
        "reasoning": "Target-oriented models designed for marketing. VMID and SCM show best performance on CollegeMsg dataset.",
        "avoid": ["irSIR", "UTM"],
        "metrics_priority": ["coverage", "temporal_accuracy"]
    },
    "competitive_marketing": {
        "top_models": ["TBCELF", "TIC", "AtI", "TICC", "IML-IC"],
        "reasoning": "Competition-oriented models handle multiple campaigns. TBCELF excels with budget constraints.",
        "avoid": ["SI", "Voter"],
        "metrics_priority": ["coverage", "competitive"]
    },
    "epidemic_outbreak": {
        "top_models": ["SIR", "SEIR", "SI", "cpSI-R", "SCM"],
        "reasoning": "Classical epidemic models with recovery mechanisms. SEIR for latency periods, cpSI-R for reinfection.",
        "avoid": ["IC", "LT"],
        "metrics_priority": ["temporal_accuracy", "behavioral"]
    },
    "antimicrobial_resistance": {
        "top_models": ["SCIR", "FSIR", "SEIR"],
        "reasoning": "Explanatory models for genetic transmission and stratified interactions.",
        "avoid": ["IC", "Voter"],
        "metrics_priority": ["behavioral", "complexity"]
    },
    "cancer_progression": {
        "top_models": ["cpSI-R", "SCIR", "FSIR", "TBCELF", "IML-IC"],
        "reasoning": "Reinfection models for treatment resistance. TBCELF for precision therapy targeting.",
        "avoid": ["MTM", "STM"],
        "metrics_priority": ["temporal_accuracy", "behavioral"]
    },
    "opinion_dynamics": {
        "top_models": ["Voter", "EVM", "ISR", "POE", "LIM"],
        "reasoning": "Interaction-oriented models capture consensus formation and social influence.",
        "avoid": ["SI", "IC"],
        "metrics_priority": ["behavioral", "temporal"]
    },
    "influencer_marketing": {
        "top_models": ["TAM", "TLRA", "LIM", "VMID"],
        "reasoning": "Group-oriented models identify influential communities and topic leaders.",
        "avoid": ["UTM", "DBM"],
        "metrics_priority": ["behavioral", "coverage"]
    },
    "misinformation_control": {
        "top_models": ["TrCID", "FST-N", "FSC-N", "DLT"],
        "reasoning": "Trust-aware and blocking models for suppressing harmful diffusion.",
        "avoid": ["STM", "MTM"],
        "metrics_priority": ["behavioral", "competitive"]
    },
    "emergency_alerts": {
        "top_models": ["STM", "IC", "SI", "WPM"],
        "reasoning": "Fast-triggering models for rapid information dissemination.",
        "avoid": ["UTM", "DVM"],
        "metrics_priority": ["coverage", "temporal"]
    },
    "political_campaigns": {
        "top_models": ["EVM", "BDVM", "ISR", "DCM"],
        "reasoning": "Opinion formation models with bias and decision-awareness.",
        "avoid": ["SI", "LND"],
        "metrics_priority": ["behavioral", "competitive"]
    },
    "product_adoption": {
        "top_models": ["BIADM", "ABBM", "PAM", "LTC"],
        "reasoning": "Structure-oriented models for market penetration and consumer behavior.",
        "avoid": ["SIR", "SEIR"],
        "metrics_priority": ["coverage", "behavioral"]
    },
    "budget_constrained": {
        "top_models": ["TBCELF", "OCM", "ASIC"],
        "reasoning": "Models optimized for cost-effective seed selection.",
        "avoid": ["CTM-IC", "TCC"],
        "metrics_priority": ["complexity", "coverage"]
    }
}

# ============================
# VALIDATION SCENARIOS FROM PAPER
# ============================

PAPER_VALIDATION = {
    "higgs_twitter_rumor": {
        "dataset": "Higgs Twitter",
        "ground_truth": "real",
        "best_performers": {
            "k=5": ["UAD", "SI", "SIR", "LND", "PAM"],
            "k=10": ["SI", "UAD", "SIR", "IC", "LND"],
            "k=15": ["UAD", "SI", "SIR", "LND", "IC"],
            "k=20": ["UAD", "SI", "MTM", "SIR", "LND"]
        },
        "worst_performers": ["ABBM", "SCM"],
        "key_insight": "irSIR achieves lowest RMSE; interaction-aware models best reproduce social rumor dynamics"
    },
    "hospital_ward_disease": {
        "dataset": "Hospital Ward",
        "ground_truth": "IC(p=1)",
        "best_performers": {
            "k=5": ["TAM", "WPM", "SI", "LT", "SEIR"],
            "k=10": ["WPM", "SI", "TAM", "SEIR", "LT"],
            "k=15": ["WPM", "SI", "SEIR", "LT", "TAM"],
            "k=20": ["TAM", "SI", "WPM", "SEIR", "SCM"]
        },
        "worst_performers": ["SIR", "UAD", "POE"],
        "key_insight": "SCM and LT achieve near-zero error; structurally driven models excel in deterministic epidemic settings"
    },
    "collegemsg_marketing": {
        "dataset": "CollegeMsg",
        "ground_truth": "IC(p=1)",
        "best_performers": {
            "k=5": ["SCM", "LT", "SI", "AtI", "MAT"],
            "k=10": ["SCM", "LT", "AtI", "SI", "MAT"],
            "k=15": ["SCM", "LT", "SI", "MAT", "SEIR"],
            "k=20": ["SCM", "LT", "SI", "AtI", "SEIR"]
        },
        "worst_performers": ["irSIR", "PAM", "IC"],
        "key_insight": "SCM consistently most accurate; target-oriented models suitable for viral marketing"
    }
}

# ============================
# WEIGHT PERSISTENCE
# ============================

WEIGHT_FILE = "diffusion_model_weights.json"

def load_weights():
    if os.path.exists(WEIGHT_FILE):
        with open(WEIGHT_FILE, "r") as f:
            return json.load(f)
    return {m: 1.0 for m in MODELS}

def save_weights(weights):
    with open(WEIGHT_FILE, "w") as f:
        json.dump(weights, f, indent=2)

MODEL_WEIGHTS = load_weights()

# ============================
# ENHANCED QUESTIONNAIRE
# ============================

def ask_questions() -> Dict[str, Any]:
    print("\n" + "="*70)
    print("  DIFFUSION MODEL RECOMMENDATION SYSTEM")
    print("  Based on: 'Diffusion Models for Influence Maximization")
    print("           on Temporal Networks: A Guide to Make the Best Choice'")
    print("="*70 + "\n")

    answers = {}
    
    # Question 1: Primary Scenario
    print("1) What is your PRIMARY application scenario?")
    print("   a) Rumor/information spreading on social media")
    print("   b) Disease/epidemic outbreak modeling")
    print("   c) Viral marketing campaign")
    print("   d) Competitive marketing (multiple brands/campaigns)")
    print("   e) Antimicrobial resistance spread")
    print("   f) Cancer progression modeling")
    print("   g) Opinion dynamics / political campaigns")
    print("   h) Influencer identification")
    print("   i) Misinformation control")
    print("   j) Emergency alert dissemination")
    print("   k) Product adoption forecasting")
    print("   l) Other (general influence maximization)")
    
    scenario_map = {
        'a': 'rumor_spreading',
        'b': 'epidemic_outbreak',
        'c': 'viral_marketing',
        'd': 'competitive_marketing',
        'e': 'antimicrobial_resistance',
        'f': 'cancer_progression',
        'g': 'political_campaigns',
        'h': 'influencer_marketing',
        'i': 'misinformation_control',
        'j': 'emergency_alerts',
        'k': 'product_adoption',
        'l': 'general'
    }
    
    choice = input("   Enter choice (a-l): ").lower()
    answers["scenario"] = scenario_map.get(choice, 'general')
    
    # Question 2: Network Type
    print("\n2) What type of network are you working with?")
    print("   a) Temporal network (time-stamped interactions)")
    print("   b) Static network (no temporal information)")
    print("   c) Dynamic network (evolving structure)")
    choice = input("   Enter choice (a-c): ").lower()
    answers["network_type"] = "temporal" if choice == 'a' else "dynamic" if choice == 'c' else "static"
    answers["temporal"] = choice in ['a', 'c']
    
    # Question 3: Ground Truth
    print("\n3) Do you have real diffusion ground truth data?")
    print("   a) Yes, empirical diffusion traces (like Higgs Twitter)")
    print("   b) No, but can use idealized model (IC with p=1)")
    print("   c) No ground truth available")
    choice = input("   Enter choice (a-c): ").lower()
    answers["ground_truth"] = "real" if choice == 'a' else "idealized" if choice == 'b' else "none"
    
    # Question 4: Primary Objective
    print("\n4) What is your PRIMARY optimization goal?")
    print("   a) Maximum accuracy (match real diffusion patterns)")
    print("   b) Maximum reach/coverage")
    print("   c) Fairness across competing entities")
    print("   d) Computational efficiency (low runtime)")
    print("   e) Cost-effectiveness (budget constraints)")
    print("   f) Early-stage prediction accuracy")
    choice = input("   Enter choice (a-f): ").lower()
    goal_map = {
        'a': 'accuracy',
        'b': 'reach',
        'c': 'fairness',
        'd': 'efficiency',
        'e': 'cost',
        'f': 'early_accuracy'
    }
    answers["goal"] = goal_map.get(choice, 'reach')
    
    # Question 5: Competition
    print("\n5) Are there competing diffusion processes?")
    print("   (e.g., multiple brands, competing diseases, adversarial influence)")
    answers["competition"] = input("   Yes/No: ").lower().startswith('y')
    
    # Question 6: Behavioral Factors
    print("\n6) Do behavioral/psychological factors significantly influence spread?")
    print("   (e.g., opinion formation, trust, preferences, decision-making)")
    answers["behavioral"] = input("   Yes/No: ").lower().startswith('y')
    
    # Question 7: Community Structure
    print("\n7) Does your network have strong community/cluster structure?")
    answers["community"] = input("   Yes/No: ").lower().startswith('y')
    
    # Question 8: Seed Selection Constraints
    print("\n8) What are your seed selection constraints?")
    print("   a) No constraints (maximize spread)")
    print("   b) Budget constraints (limited resources)")
    print("   c) Fairness constraints (balanced across groups)")
    print("   d) Targeted intervention (specific high-value nodes)")
    choice = input("   Enter choice (a-d): ").lower()
    constraint_map = {
        'a': 'none',
        'b': 'budget',
        'c': 'fairness',
        'd': 'targeted'
    }
    answers["seed_constraint"] = constraint_map.get(choice, 'none')
    answers["budget"] = choice == 'b'
    
    # Question 9: Temporal Dynamics
    print("\n9) What temporal characteristics are important?")
    print("   a) Early-stage accuracy (initial spread)")
    print("   b) Long-term saturation (final coverage)")
    print("   c) Peak timing (when maximum spread occurs)")
    print("   d) Entire temporal trajectory")
    choice = input("   Enter choice (a-d): ").lower()
    timing_map = {
        'a': 'early',
        'b': 'final',
        'c': 'peak',
        'd': 'full'
    }
    answers["timing_priority"] = timing_map.get(choice, 'full')
    answers["speed"] = choice in ['a', 'c']
    
    # Question 10: Model Properties
    print("\n10) Do you require guaranteed approximation bounds?")
    print("    (i.e., submodular + monotone properties for greedy algorithms)")
    answers["require_guarantees"] = input("    Yes/No: ").lower().startswith('y')
    
    # Question 11: Network Size
    print("\n11) What is your network size?")
    print("    a) Small (<1,000 nodes)")
    print("    b) Medium (1,000-10,000 nodes)")
    print("    c) Large (10,000-100,000 nodes)")
    print("    d) Very large (>100,000 nodes)")
    choice = input("    Enter choice (a-d): ").lower()
    size_map = {
        'a': 'small',
        'b': 'medium',
        'c': 'large',
        'd': 'very_large'
    }
    answers["network_size"] = size_map.get(choice, 'medium')
    
    # Question 12: Special Features
    print("\n12) Does your scenario involve any of these special features?")
    print("    a) Reinfection / reactivation dynamics")
    print("    b) Recovery mechanisms")
    print("    c) Latency / incubation periods")
    print("    d) Trust / distrust relationships")
    print("    e) Memory effects (past attempts matter)")
    print("    f) None of the above")
    choice = input("    Enter choice (a-f): ").lower()
    feature_map = {
        'a': 'reinfection',
        'b': 'recovery',
        'c': 'latency',
        'd': 'trust',
        'e': 'memory',
        'f': 'none'
    }
    answers["special_feature"] = feature_map.get(choice, 'none')
    
    # Question 13: Learning
    print("\n13) Should the recommendation system learn from your feedback?")
    answers["learning"] = input("    Yes/No: ").lower().startswith('y')
    
    return answers

# ============================
# ADVANCED SCORING ENGINE
# ============================

def score_models(answers: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """
    Score models based on answers using multi-factor weighted scoring.
    Returns: List of (model_name, score, reasoning)
    """
    scores = defaultdict(lambda: {"score": 0.0, "reasons": []})
    
    # Get scenario-specific recommendations if available
    scenario = answers.get("scenario", "general")
    scenario_rec = SCENARIO_RECOMMENDATIONS.get(scenario, {})
    
    for model, props in MODELS.items():
        base_score = MODEL_WEIGHTS.get(model, 1.0)
        reasons = []
        
        # Scenario match (highest weight)
        if model in scenario_rec.get("top_models", []):
            base_score += 5.0
            reasons.append(f"Top recommendation for {scenario}")
        elif model in scenario_rec.get("avoid", []):
            base_score -= 3.0
            reasons.append(f"Not recommended for {scenario}")
        
        # Temporal network requirement
        if answers.get("temporal") and props.get("temporal"):
            base_score += 2.5
            reasons.append("Supports temporal dynamics")
        elif answers.get("temporal") and not props.get("temporal"):
            base_score -= 2.0
        
        # Competition requirement
        if answers.get("competition") and props.get("competitive"):
            base_score += 3.5
            reasons.append("Handles competitive diffusion")
        elif answers.get("competition") and not props.get("competitive"):
            base_score -= 1.5
        
        # Behavioral factors
        if answers.get("behavioral") and props.get("behavioral"):
            base_score += 3.0
            reasons.append("Incorporates behavioral dynamics")
        
        # Network size considerations
        network_size = answers.get("network_size", "medium")
        if network_size == "very_large":
            if props.get("complexity") == "Polynomial":
                base_score += 2.0
                reasons.append("Scalable (polynomial complexity)")
            elif "ASIC" in model or "TBCELF" in model:
                base_score += 1.5
                reasons.append("Optimized for large networks")
        
        # Goal-specific scoring
        goal = answers.get("goal", "reach")
        
        if goal == "accuracy":
            if props.get("temporal_accuracy") == "very_high":
                base_score += 4.0
                reasons.append("Very high temporal accuracy")
            elif props.get("temporal_accuracy") == "high":
                base_score += 2.5
                reasons.append("High temporal accuracy")
        
        if goal == "reach":
            if props.get("coverage") == "very_high":
                base_score += 4.0
                reasons.append("Very high coverage")
            elif props.get("coverage") == "high":
                base_score += 2.5
                reasons.append("High coverage")
        
        if goal == "efficiency":
            if props.get("complexity") == "Polynomial":
                base_score += 3.5
                reasons.append("Computationally efficient")
        
        if goal == "cost" and answers.get("budget"):
            if model == "TBCELF":
                base_score += 5.0
                reasons.append("★ BEST for budget-constrained scenarios")
            elif "budget" in str(props.get("best_for", [])).lower():
                base_score += 2.0
        
        if goal == "early_accuracy" and answers.get("speed"):
            if model in ["irSIR", "IC", "TIC", "STM"]:
                base_score += 3.0
                reasons.append("Excellent for early-stage prediction")
        
        # Special feature matching
        special = answers.get("special_feature", "none")
        if special == "reinfection" and "reinfection" in str(props.get("best_for", [])).lower():
            base_score += 3.0
            reasons.append("Handles reinfection dynamics")
        if special == "recovery" and "recovery" in model.lower():
            base_score += 2.5
            reasons.append("Includes recovery mechanisms")
        if special == "latency" and "SEIR" in model:
            base_score += 3.0
            reasons.append("Models latency periods")
        if special == "trust" and "trust" in str(props).lower():
            base_score += 3.0
            reasons.append("Trust-aware model")
        if special == "memory" and "TCC" in model:
            base_score += 3.5
            reasons.append("Captures memory effects")
        
        # Guarantee requirements
        if answers.get("require_guarantees"):
            if props.get("submodular") and props.get("monotone"):
                base_score += 2.0
                reasons.append("Provides approximation guarantees")
            else:
                base_score -= 1.5
        
        # Community structure
        if answers.get("community"):
            if "community" in str(props.get("best_for", [])).lower() or \
               "cluster" in str(props.get("best_for", [])).lower():
                base_score += 2.0
                reasons.append("Leverages community structure")
        
        # Seed constraint matching
        seed_constraint = answers.get("seed_constraint", "none")
        if seed_constraint == "targeted" and model in ["TBCELF", "IML-IC", "TIC"]:
            base_score += 2.5
            reasons.append("Optimized for targeted intervention")
        
        # Paper validation boost
        ground_truth = answers.get("ground_truth", "none")
        if ground_truth in ["real", "idealized"]:
            for validation_scenario, data in PAPER_VALIDATION.items():
                if scenario in validation_scenario or answers.get("scenario") in validation_scenario:
                    for k_size, best_models in data.get("best_performers", {}).items():
                        if model in best_models:
                            base_score += 3.0
                            reasons.append(f"★ Validated: Top performer on {data['dataset']}")
                            break
                    if model in data.get("worst_performers", []):
                        base_score -= 2.0
                        reasons.append(f"⚠ Poor performer on {data['dataset']}")
        
        scores[model]["score"] = base_score
        scores[model]["reasons"] = reasons
    
    # Sort by score
    ranked = sorted(
        [(model, data["score"], "; ".join(data["reasons"])) 
         for model, data in scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked

# ============================
# DETAILED RECOMMENDATIONS
# ============================

def display_recommendations(ranked: List[Tuple[str, float, str]], answers: Dict[str, Any]):
    """Display top recommendations with detailed explanations"""
    
    scenario = answers.get("scenario", "general")
    scenario_rec = SCENARIO_RECOMMENDATIONS.get(scenario, {})
    
    print("\n" + "="*70)
    print("  TOP RECOMMENDED MODELS")
    print("="*70)
    
    # Show scenario-specific insight
    if scenario_rec:
        print(f"\nScenario: {scenario.replace('_', ' ').title()}")
        print(f"Expert Insight: {scenario_rec.get('reasoning', 'N/A')}")
        print()
    
    # Display top 7 recommendations
    for i, (model, score, reasoning) in enumerate(ranked[:7], 1):
        props = MODELS[model]
        
        print(f"\n{i}. {model}")
        print(f"   Score: {score:.2f}")
        print(f"   Category: {props['taxonomy']}")
        print(f"   Description: {props['description']}")
        
        if reasoning:
            print(f"   Why recommended: {reasoning}")
        
        # Show key properties
        properties = []
        if props.get("temporal"):
            properties.append("Temporal")
        if props.get("competitive"):
            properties.append("Competitive")
        if props.get("behavioral"):
            properties.append("Behavioral")
        if props.get("submodular") and props.get("monotone"):
            properties.append("Guarantees ✓")
        
        if properties:
            print(f"   Properties: {', '.join(properties)}")
        
        print(f"   Complexity: {props.get('complexity', 'Unknown')}")
        print(f"   Coverage: {props.get('coverage', 'Unknown').title()}")
        print(f"   Temporal Accuracy: {props.get('temporal_accuracy', 'Unknown').title()}")
        
        print("   " + "-"*66)
    
    # Show paper validation if applicable
    print("\n" + "="*70)
    print("  VALIDATION FROM PAPER")
    print("="*70)
    
    validation_shown = False
    for validation_scenario, data in PAPER_VALIDATION.items():
        if scenario in validation_scenario or answers.get("scenario") in validation_scenario:
            print(f"\nDataset: {data['dataset']}")
            print(f"Ground Truth: {data['ground_truth']}")
            print(f"Key Insight: {data['key_insight']}")
            print("\nBest Performers by Seed Size:")
            for k_size, models in data.get("best_performers", {}).items():
                print(f"  {k_size}: {', '.join(models[:3])}")
            validation_shown = True
    
    if not validation_shown:
        print("\nNo specific paper validation available for this scenario.")
        print("Recommendations based on model properties and expert knowledge.")
    
    # Show models to avoid
    if scenario_rec.get("avoid"):
        print("\n" + "="*70)
        print("  MODELS TO AVOID FOR THIS SCENARIO")
        print("="*70)
        print(f"\n{', '.join(scenario_rec['avoid'])}")
        print(f"\nReason: These models showed poor performance or are not")
        print(f"suitable for {scenario.replace('_', ' ')} scenarios.")

# ============================
# FEEDBACK LEARNING
# ============================

def collect_feedback(top_model: str, answers: Dict[str, Any]):
    """Enhanced feedback collection with multiple dimensions"""
    
    if not answers.get("learning", False):
        return
    
    print("\n" + "="*70)
    print("  FEEDBACK")
    print("="*70)
    
    print(f"\nYou selected: {top_model}")
    print("Please provide feedback to improve future recommendations:\n")
    
    # Performance feedback
    print("1) How well did this model perform?")
    print("   a) Excellent - exceeded expectations")
    print("   b) Good - met expectations")
    print("   c) Fair - acceptable but could be better")
    print("   d) Poor - did not meet needs")
    perf = input("   Enter choice (a-d): ").lower()
    
    perf_weight = {'a': 0.8, 'b': 0.3, 'c': -0.2, 'd': -0.5}.get(perf, 0.0)
    
    # Specific aspects
    if perf in ['a', 'b']:
        print("\n2) What worked particularly well? (select multiple: e.g., a,c)")
        print("   a) Temporal accuracy")
        print("   b) Coverage/reach")
        print("   c) Computational efficiency")
        print("   d) Ease of implementation")
        print("   e) Model interpretability")
        aspects = input("   Enter choices: ").lower().split(',')
        
        # Boost related properties
        if 'a' in aspects:
            MODEL_WEIGHTS[top_model] += 0.3
        if 'b' in aspects:
            MODEL_WEIGHTS[top_model] += 0.3
        if 'c' in aspects and MODELS[top_model].get("complexity") == "Polynomial":
            MODEL_WEIGHTS[top_model] += 0.2
    
    elif perf in ['c', 'd']:
        print("\n2) What was the main issue?")
        print("   a) Poor temporal accuracy")
        print("   b) Low coverage")
        print("   c) Too computationally expensive")
        print("   d) Difficult to implement")
        print("   e) Results didn't match expectations")
        issue = input("   Enter choice (a-e): ").lower()
        
        if issue in ['a', 'b', 'e']:
            perf_weight -= 0.3
    
    # Update weights
    MODEL_WEIGHTS[top_model] += perf_weight
    
    # Prevent negative weights
    if MODEL_WEIGHTS[top_model] < 0.1:
        MODEL_WEIGHTS[top_model] = 0.1
    
    save_weights(MODEL_WEIGHTS)
    
    print("\n✓ Feedback recorded. The system will provide better recommendations over time.")

# ============================
# EXPORT FUNCTIONALITY
# ============================

def export_recommendations(ranked: List[Tuple[str, float, str]], answers: Dict[str, Any]):
    """Export recommendations to a JSON file"""
    
    export_data = {
        "scenario": answers.get("scenario"),
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "user_answers": answers,
        "recommendations": [
            {
                "rank": i,
                "model": model,
                "score": score,
                "reasoning": reasoning,
                "properties": MODELS[model]
            }
            for i, (model, score, reasoning) in enumerate(ranked[:10], 1)
        ]
    }
    
    filename = f"recommendations_{answers.get('scenario', 'general')}.json"
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n✓ Recommendations exported to: {filename}")

# ============================
# MAIN APPLICATION
# ============================

def main():
    """Main application flow"""
    
    try:
        # Collect user inputs
        answers = ask_questions()
        
        # Score and rank models
        ranked = score_models(answers)
        
        # Display recommendations
        display_recommendations(ranked, answers)
        
        # Ask if user wants to export
        print("\n" + "="*70)
        export = input("\nWould you like to export these recommendations? (yes/no): ")
        if export.lower().startswith('y'):
            export_recommendations(ranked, answers)
        
        # Collect feedback
        if ranked:
            top_model = ranked[0][0]
            collect_feedback(top_model, answers)
        
        print("\n" + "="*70)
        print("  Thank you for using the Diffusion Model Recommender!")
        print("="*70 + "\n")
    
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    main()
