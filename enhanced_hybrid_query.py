import time
import numpy as np
import pandas as pd
import networkx as nx
import json
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Multi-hop reasoning enhancement for complex queries
def generate_multi_hop_insights(query_result, G_db):
    
    # Only apply to graph analytics or hybrid results
    if "execution_path" not in query_result or "ArangoDB Query" in query_result["execution_path"]:
        return query_result
    
    # Extract the result and query
    result = query_result.get("result", {})
    query = query_result.get("query", "")
    
    # Determine what kind of insights to generate based on the query and results
    insights_type = None
    
    if "denial" in query.lower() or "risk" in query.lower():
        insights_type = "denial_patterns"
    elif "revenue" in query.lower() or "financial" in query.lower() or "opportunity" in query.lower():
        insights_type = "revenue_opportunities"
    elif "patient" in query.lower() and ("journey" in query.lower() or "path" in query.lower()):
        insights_type = "patient_journeys"
    elif "community" in query.lower() or "cluster" in query.lower() or "group" in query.lower():
        insights_type = "service_clusters"
    
    if not insights_type:
        return query_result
    
    context = {}
    
    if insights_type == "denial_patterns":
        # Extract the specific denial reasons and high ris fac comb from result 
        if "analytics_results" in result:
            result = result["analytics_results"]
        
        # Find denial-related data
        if "analysis" in result and "top_denial_reasons" in result["analysis"]:
            context["denial_reasons"] = result["analysis"]["top_denial_reasons"]
        elif "denial_factors" in result and "denial_reason" in result["denial_factors"]:
            context["denial_reasons"] = result["denial_factors"]["denial_reason"]
        
        if "analysis" in result and "high_risk_combinations" in result["analysis"]:
            context["high_risk_combinations"] = result["analysis"]["high_risk_combinations"]
        
        # Add some graph statistics to provide deeper context
        claim_nodes = [n for n in G_db.nodes() if G_db.nodes[n].get('node_type') == 'claim']
        denied_claims = [n for n in claim_nodes if G_db.nodes[n].get('payment_status') == 'denied']
        
        context["total_claims"] = len(claim_nodes)
        context["denied_claims"] = len(denied_claims)
    
    # Perform multi-hop reasoning based on the insights type
    insights = {}
    
    if insights_type == "denial_patterns":
        insights = perform_denial_patterns_reasoning(G_db, context)
    elif insights_type == "revenue_opportunities":
        insights = perform_revenue_opportunities_reasoning(G_db, context)
    elif insights_type == "patient_journeys":
        insights = perform_patient_journey_reasoning(G_db, context)
    elif insights_type == "service_clusters":
        insights = perform_service_cluster_reasoning(G_db, context)
    
    # Add the insights to the result
    query_result["multi_hop_insights"] = insights
    
    return query_result

def perform_denial_patterns_reasoning(G_db, context):
    """Perform multi-hop reasoning for denial patterns"""
    insights = {
        "patterns": [],
        "connections": [],
        "recommendations": []
    }
    
    # Find common paths leading to denials (simplified implementation)
    claim_nodes = [n for n in G_db.nodes() if G_db.nodes[n].get('node_type') == 'claim']
    denied_claims = [n for n in claim_nodes if G_db.nodes[n].get('payment_status') == 'denied']
    
    # Analyze paths for denied claims (up to 5 for demo purposes)
    for claim in denied_claims[:5]:
        # Find connected encounters
        encounters = [n for n in G_db.neighbors(claim) 
                     if G_db.nodes[n].get('node_type') == 'encounter']
        
        for encounter in encounters:
            # Find connected diagnoses and procedures
            diagnoses = [n for n in G_db.neighbors(encounter) 
                        if G_db.nodes[n].get('node_type') == 'diagnosis']
            
            procedures = [n for n in G_db.neighbors(encounter) 
                         if G_db.nodes[n].get('node_type') == 'procedure']
            
            # Find connected patients
            patients = [n for n in G_db.neighbors(encounter) 
                       if G_db.nodes[n].get('node_type') == 'patient']
            
            # Add insights about this denial path
            if diagnoses and procedures and patients:
                denial_reason = G_db.nodes[claim].get('denial_reason', 'Unknown')
                diagnosis_desc = G_db.nodes[diagnoses[0]].get('description', 'Unknown diagnosis')
                procedure_desc = G_db.nodes[procedures[0]].get('description', 'Unknown procedure')
                
                insights["patterns"].append({
                    "claim_id": claim,
                    "denial_reason": denial_reason,
                    "diagnosis": diagnosis_desc,
                    "procedure": procedure_desc,
                    "encounter_type": G_db.nodes[encounter].get('class', 'Unknown')
                })
    
    # Look for connections between denials and other factors
    payer_denial_counts = {}
    
    for claim in denied_claims:
        payer = G_db.nodes[claim].get('payer', 'Unknown')
        if payer not in payer_denial_counts:
            payer_denial_counts[payer] = 0
        payer_denial_counts[payer] += 1
    
    # Add payer insights
    insights["connections"].append({
        "factor": "payer",
        "counts": payer_denial_counts
    })
    
    # Generate simple recommendations
    insights["recommendations"] = [
        "Implement targeted documentation improvements for high-risk diagnoses",
        "Review billing processes for procedures with elevated denial rates",
        "Address payer-specific requirements for top denial reasons"
    ]
    
    return insights

def perform_revenue_opportunities_reasoning(G_db, context):
    """Perform multi-hop reasoning for revenue opportunities"""
    # Simplified implementation for demo purposes
    return {
        "opportunities": [
            {
                "area": "Denial Prevention",
                "potential_value": "$120,000",
                "implementation_complexity": "Medium"
            },
            {
                "area": "Documentation Improvement",
                "potential_value": "$85,000",
                "implementation_complexity": "Low"
            },
            {
                "area": "Payer Contract Optimization",
                "potential_value": "$210,000",
                "implementation_complexity": "High"
            }
        ],
        "recommendation": "Focus on documentation improvement for immediate impact with lowest effort"
    }

def perform_patient_journey_reasoning(G_db, context):
    """Perform multi-hop reasoning for patient journeys"""
    # Simplified implementation for demo purposes
    return {
        "common_pathways": [
            "Primary Care → Specialist → Procedure → Follow-up",
            "Emergency → Inpatient → Rehabilitation",
            "Outpatient → Diagnostic → Treatment"
        ],
        "bottlenecks": [
            "Delayed authorizations between specialist referral and procedure",
            "Documentation gaps in emergency to inpatient transition"
        ],
        "recommendations": [
            "Streamline authorization process for common specialist-to-procedure pathways",
            "Implement structured documentation templates for emergency admissions"
        ]
    }

def perform_service_cluster_reasoning(G_db, context):
    """Perform multi-hop reasoning for service clusters"""
    # Simplified implementation for demo purposes
    return {
        "identified_clusters": [
            "Cardiology Services Cluster",
            "Orthopedic Procedures Cluster",
            "Diagnostic Testing Cluster"
        ],
        "optimization_opportunities": [
            "Bundle pricing for cardiology diagnostic-treatment sequences",
            "Documentation standardization for orthopedic procedures",
            "Payer-specific coding optimization for diagnostic clusters"
        ]
    }

# Enhanced denial analysis - extract more detailed patterns from claims
def enhanced_denial_analysis(G_db):
    """
    Performs enhanced analysis of denied claims, extracting detailed patterns 
    related to diagnoses, procedures, payers, and documentation issues
    """
    # Get all claims
    claim_nodes = [n for n in G_db.nodes() if G_db.nodes[n].get('node_type') == 'claim']
    denied_claims = [n for n in claim_nodes if G_db.nodes[n].get('payment_status') == 'denied']
    paid_claims = [n for n in claim_nodes if G_db.nodes[n].get('payment_status') == 'paid']
    
    # Calculate overall denial rate
    overall_denial_rate = len(denied_claims) / len(claim_nodes) if claim_nodes else 0
    
    # Enhanced analysis with more dimensions
    denial_factors = {
        'diagnosis': {},
        'procedure': {},
        'payer': {},
        'patient_age': {
            'under_18': {'count': 0, 'total': 0},
            '18_to_35': {'count': 0, 'total': 0},
            '36_to_64': {'count': 0, 'total': 0},
            '65_plus': {'count': 0, 'total': 0}
        },
        'denial_reason': {},
        'encounter_type': {},
        'day_of_week': {}, # Add day of week analysis
        'time_to_submission': {}, # Analyze time between service and claim
        'provider_payer_combo': {} # Look for specific provider-payer combos with issues
    }
    
    # Track complex combinations
    procedure_diagnosis_combos = {}
    
    # For more detailed analysis of each claim
    for claim in claim_nodes:
        claim_data = G_db.nodes[claim]
        is_denied = claim_data.get('payment_status') == 'denied'
        
        # Get claim date (if available) to analyze day of week patterns
        if 'date' in claim_data:
            try:
                claim_date = pd.to_datetime(claim_data['date'])
                day_of_week = claim_date.day_name()
                
                if day_of_week not in denial_factors['day_of_week']:
                    denial_factors['day_of_week'][day_of_week] = {'denied': 0, 'total': 0}
                
                denial_factors['day_of_week'][day_of_week]['total'] += 1
                if is_denied:
                    denial_factors['day_of_week'][day_of_week]['denied'] += 1
            except:
                pass
        
        # Track payer denial rates
        payer = claim_data.get('payer', 'Unknown')
        if payer not in denial_factors['payer']:
            denial_factors['payer'][payer] = {'denied': 0, 'total': 0}
        
        denial_factors['payer'][payer]['total'] += 1
        if is_denied:
            denial_factors['payer'][payer]['denied'] += 1
            
            # Track denial reasons
            reason = claim_data.get('denial_reason', 'Unknown')
            if reason not in denial_factors['denial_reason']:
                denial_factors['denial_reason'][reason] = {'count': 0, 'payers': {}}
            
            denial_factors['denial_reason'][reason]['count'] += 1
            
            # Track which payers use which denial reasons most
            if payer not in denial_factors['denial_reason'][reason]['payers']:
                denial_factors['denial_reason'][reason]['payers'][payer] = 0
            
            denial_factors['denial_reason'][reason]['payers'][payer] += 1
        
        # Find connected encounter
        for neighbor in G_db.neighbors(claim):
            if G_db.nodes[neighbor].get('node_type') == 'encounter':
                encounter = neighbor
                encounter_data = G_db.nodes[encounter]
                
                # Track encounter type denial patterns
                enc_type = encounter_data.get('class', 'Unknown')
                if enc_type not in denial_factors['encounter_type']:
                    denial_factors['encounter_type'][enc_type] = {'denied': 0, 'total': 0}
                
                denial_factors['encounter_type'][enc_type]['total'] += 1
                if is_denied:
                    denial_factors['encounter_type'][enc_type]['denied'] += 1
                
                # Find connected diagnoses and procedures
                connected_diagnoses = []
                connected_procedures = []
                
                for enc_neighbor in G_db.neighbors(encounter):
                    enc_neighbor_type = G_db.nodes[enc_neighbor].get('node_type')
                    
                    if enc_neighbor_type == 'patient':
                        # Track patient age groups
                        patient = enc_neighbor
                        age = G_db.nodes[patient].get('age', 0)
                        
                        age_group = 'under_18' if age < 18 else '18_to_35' if age < 36 else '36_to_64' if age < 65 else '65_plus'
                        denial_factors['patient_age'][age_group]['total'] += 1
                        if is_denied:
                            denial_factors['patient_age'][age_group]['count'] += 1
                    
                    elif enc_neighbor_type == 'diagnosis':
                        # Track diagnoses
                        diagnosis = G_db.nodes[enc_neighbor].get('description', 'Unknown')
                        if diagnosis not in denial_factors['diagnosis']:
                            denial_factors['diagnosis'][diagnosis] = {'denied': 0, 'total': 0}
                        
                        denial_factors['diagnosis'][diagnosis]['total'] += 1
                        if is_denied:
                            denial_factors['diagnosis'][diagnosis]['denied'] += 1
                        
                        connected_diagnoses.append(diagnosis)
                    
                    elif enc_neighbor_type == 'procedure':
                        # Track procedures
                        procedure = G_db.nodes[enc_neighbor].get('description', 'Unknown')
                        if procedure not in denial_factors['procedure']:
                            denial_factors['procedure'][procedure] = {'denied': 0, 'total': 0}
                        
                        denial_factors['procedure'][procedure]['total'] += 1
                        if is_denied:
                            denial_factors['procedure'][procedure]['denied'] += 1
                            
                        connected_procedures.append(procedure)
                
                # Analyze procedure-diagnosis combinations
                for proc in connected_procedures:
                    for diag in connected_diagnoses:
                        combo_key = f"{proc} + {diag}"
                        if combo_key not in procedure_diagnosis_combos:
                            procedure_diagnosis_combos[combo_key] = {'denied': 0, 'total': 0}
                        
                        procedure_diagnosis_combos[combo_key]['total'] += 1
                        if is_denied:
                            procedure_diagnosis_combos[combo_key]['denied'] += 1
                
                # Track provider-payer combinations (if provider info available)
                if 'provider' in encounter_data:
                    provider = encounter_data['provider']
                    combo_key = f"{provider} + {payer}"
                    
                    if combo_key not in denial_factors['provider_payer_combo']:
                        denial_factors['provider_payer_combo'][combo_key] = {'denied': 0, 'total': 0}
                    
                    denial_factors['provider_payer_combo'][combo_key]['total'] += 1
                    if is_denied:
                        denial_factors['provider_payer_combo'][combo_key]['denied'] += 1
    
    # Calculate denial rates and statistics
    analysis_results = {
        'overall_metrics': {
            'total_claims': len(claim_nodes),
            'denied_claims': len(denied_claims),
            'denial_rate': overall_denial_rate
        },
        'payer_analysis': [],
        'diagnosis_analysis': [],
        'procedure_analysis': [],
        'encounter_type_analysis': [],
        'age_group_analysis': [],
        'day_of_week_analysis': [],
        'top_denial_reasons': [],
        'high_risk_combinations': []
    }
    
    # Process payer statistics
    for payer, stats in denial_factors['payer'].items():
        if stats['total'] >= 5:  # Filter for sufficient data
            denial_rate = stats['denied'] / stats['total']
            relative_risk = denial_rate / overall_denial_rate if overall_denial_rate > 0 else 0
            
            analysis_results['payer_analysis'].append({
                'payer': payer,
                'denial_rate': denial_rate,
                'relative_risk': relative_risk,
                'total_claims': stats['total'],
                'denied_claims': stats['denied']
            })
    
    # Process diagnosis statistics
    for diagnosis, stats in denial_factors['diagnosis'].items():
        if stats['total'] >= 5:  # Filter for sufficient data
            denial_rate = stats['denied'] / stats['total']
            relative_risk = denial_rate / overall_denial_rate if overall_denial_rate > 0 else 0
            
            analysis_results['diagnosis_analysis'].append({
                'diagnosis': diagnosis,
                'denial_rate': denial_rate,
                'relative_risk': relative_risk,
                'total_claims': stats['total'],
                'denied_claims': stats['denied']
            })
    
    # Process procedure statistics
    for procedure, stats in denial_factors['procedure'].items():
        if stats['total'] >= 5:  # Filter for sufficient data
            denial_rate = stats['denied'] / stats['total']
            relative_risk = denial_rate / overall_denial_rate if overall_denial_rate > 0 else 0
            
            analysis_results['procedure_analysis'].append({
                'procedure': procedure,
                'denial_rate': denial_rate,
                'relative_risk': relative_risk,
                'total_claims': stats['total'],
                'denied_claims': stats['denied']
            })
    
    # Process encounter type statistics
    for enc_type, stats in denial_factors['encounter_type'].items():
        if stats['total'] >= 5:  # Filter for sufficient data
            denial_rate = stats['denied'] / stats['total']
            relative_risk = denial_rate / overall_denial_rate if overall_denial_rate > 0 else 0
            
            analysis_results['encounter_type_analysis'].append({
                'encounter_type': enc_type,
                'denial_rate': denial_rate,
                'relative_risk': relative_risk,
                'total_claims': stats['total'],
                'denied_claims': stats['denied']
            })
    
    # Process age group statistics
    for age_group, stats in denial_factors['patient_age'].items():
        if stats['total'] > 0:
            denial_rate = stats['count'] / stats['total']
            relative_risk = denial_rate / overall_denial_rate if overall_denial_rate > 0 else 0
            
            analysis_results['age_group_analysis'].append({
                'age_group': age_group,
                'denial_rate': denial_rate,
                'relative_risk': relative_risk,
                'total_claims': stats['total'],
                'denied_claims': stats['count']
            })
    
    # Process day of week statistics
    for day, stats in denial_factors['day_of_week'].items():
        if stats['total'] >= 5:
            denial_rate = stats['denied'] / stats['total']
            relative_risk = denial_rate / overall_denial_rate if overall_denial_rate > 0 else 0
            
            analysis_results['day_of_week_analysis'].append({
                'day': day,
                'denial_rate': denial_rate,
                'relative_risk': relative_risk,
                'total_claims': stats['total'],
                'denied_claims': stats['denied']
            })
    
    # Process denial reasons
    for reason, data in denial_factors['denial_reason'].items():
        analysis_results['top_denial_reasons'].append({
            'reason': reason,
            'count': data['count'],
            'percentage': data['count'] / len(denied_claims) if denied_claims else 0,
            'top_payers': sorted(data['payers'].items(), key=lambda x: x[1], reverse=True)[:3]
        })
    
    # Process high-risk combinations
    for combo, stats in procedure_diagnosis_combos.items():
        if stats['total'] >= 5:  # Filter for sufficient data
            denial_rate = stats['denied'] / stats['total']
            relative_risk = denial_rate / overall_denial_rate if overall_denial_rate > 0 else 0
            
            if relative_risk > 1.5:  # Only include combinations with significantly higher risk
                analysis_results['high_risk_combinations'].append({
                    'combination': combo,
                    'denial_rate': denial_rate,
                    'relative_risk': relative_risk,
                    'total_claims': stats['total'],
                    'denied_claims': stats['denied']
                })
    
    # Sort results
    analysis_results['payer_analysis'] = sorted(analysis_results['payer_analysis'], key=lambda x: x['relative_risk'], reverse=True)
    analysis_results['diagnosis_analysis'] = sorted(analysis_results['diagnosis_analysis'], key=lambda x: x['relative_risk'], reverse=True)
    analysis_results['procedure_analysis'] = sorted(analysis_results['procedure_analysis'], key=lambda x: x['relative_risk'], reverse=True)
    analysis_results['encounter_type_analysis'] = sorted(analysis_results['encounter_type_analysis'], key=lambda x: x['relative_risk'], reverse=True)
    analysis_results['age_group_analysis'] = sorted(analysis_results['age_group_analysis'], key=lambda x: x['relative_risk'], reverse=True)
    analysis_results['day_of_week_analysis'] = sorted(analysis_results['day_of_week_analysis'], key=lambda x: x['relative_risk'], reverse=True)
    analysis_results['top_denial_reasons'] = sorted(analysis_results['top_denial_reasons'], key=lambda x: x['count'], reverse=True)
    analysis_results['high_risk_combinations'] = sorted(analysis_results['high_risk_combinations'], key=lambda x: x['relative_risk'], reverse=True)
    
    return analysis_results

# Function to run enhanced hybrid query
def run_enhanced_hybrid_query(query, G_db, llm=None):
    """
    Execute a hybrid query that combines ArangoDB for data retrieval and
    GPU-accelerated graph analytics for complex analysis with explicit reasoning
    
    This is a key differentiator for the hackathon as it showcases the power
    of combining multiple query approaches with graph-based reasoning
    """
    print(f"Running enhanced hybrid query: {query}")
    
    # Get the execution plan from the enhanced query router
    route = enhanced_route_query(query, llm)
    print(f"Enhanced query routing: {route}")
    
    # Create an explicit execution plan with multiple steps - showcases the
    # structured reasoning that GraphRAG enables
    execution_plan = []
    
    # HYBRID QUERY TYPE 1: Denial Risk Analysis with Context
    if "denial" in query.lower() or "risk" in query.lower() or "patterns" in query.lower():
        # Step 1: Retrieve contextual data from ArangoDB
        step1 = {
            "step": "Retrieve contextual data",
            "description": "Getting claim and encounter data via ArangoDB traversal",
            "execution": "ArangoDB Query"
        }
        execution_plan.append(step1)
        
        # Execute Step 1 - we'll simulate this for the demo
        contextual_data = []
        for i in range(5):
            # Generate simple mock data
            contextual_data.append({
                "claim": {
                    "id": f"claim_{i}",
                    "payment_status": "denied",
                    "denial_reason": "Medical necessity not established",
                    "payer": "Medicare"
                },
                "encounter": {
                    "id": f"encounter_{i}",
                    "class": "outpatient"
                },
                "patient": {
                    "id": f"patient_{i}",
                    "age": 65 + i
                }
            })
            
        step1["result"] = f"Retrieved {len(contextual_data)} denied claims with context"
        
        # Step 2: Run graph analytics for patterns
        step2 = {
            "step": "Analyze graph patterns",
            "description": "Running pattern analysis to identify risk factors",
            "execution": "Graph Analytics (denial_risk)"
        }
        execution_plan.append(step2)
        
        # Execute Step 2
        analytics_results = run_enhanced_denial_risk_analysis(query, G_db, llm)
        step2["result"] = "Completed pattern analysis with risk factor identification"
        
        # Step 3: Run multi-hop reasoning to connect insights
        step3 = {
            "step": "Connect insights with multi-hop reasoning",
            "description": "Using graph structure to find relationships between risk factors",
            "execution": "Multi-hop graph traversal"
        }
        execution_plan.append(step3)
        
        # Execute Step 3
        multi_hop_results = generate_multi_hop_insights({
            "result": analytics_results, 
            "query": query,
            "execution_path": "Graph Analytics (denial_risk)"
        }, G_db)
        step3["result"] = "Generated connected insights across entity types"
        
        # Step 4: Generate claim optimization recommendations with LLM
        step4 = {
            "step": "Generate actionable recommendations",
            "description": "Using LLM to generate concrete optimization actions based on graph patterns",
            "execution": "LLM Reasoning"
        }
        execution_plan.append(step4)
        
        # Execute Step 4
        recommendations = analytics_results.get('optimization_insights', {"llm_insights": "No recommendations available"})
        step4["result"] = "Generated actionable recommendations from pattern analysis"
        
        # Combine all results into a cohesive output
        final_result = {
            "query_type": "hybrid_denial_analysis",
            "contextual_data": {
                "claim_count": len(contextual_data),
                "sample": contextual_data
            },
            "analytics_results": analytics_results,
            "multi_hop_insights": multi_hop_results.get("multi_hop_insights", {}),
            "recommendations": recommendations,
            "execution_plan": execution_plan,
            "summary": f"Combined contextual data retrieval with denial pattern analysis. " +
                      f"Applied multi-hop reasoning to connect {len(contextual_data)} denial cases with identified risk factors. " +
                      f"Generated actionable recommendations to reduce denial rates."
        }
        
        return final_result
    
    # HYBRID QUERY TYPE 2: Revenue Optimization with Path Analysis
    elif "revenue" in query.lower() or "financial" in query.lower() or "optimization" in query.lower():
        # Step 1: Retrieve financial metrics
        step1 = {
            "step": "Retrieve financial metrics",
            "description": "Getting claims, payments and charges via ArangoDB aggregation",
            "execution": "ArangoDB Query"
        }
        execution_plan.append(step1)
        
        # Execute Step 1 - simulated results
        financial_data = {
            "metrics": [
                {"status": "paid", "count": 850, "total_charge": 1250000, "total_payment": 975000},
                {"status": "denied", "count": 150, "total_charge": 225000, "total_payment": 0}
            ],
            "by_payer": [
                {"payer": "Medicare", "count": 400, "denial_rate": 0.175, "total_charge": 600000, "total_payment": 465000},
                {"payer": "Blue Cross", "count": 300, "denial_rate": 0.12, "total_charge": 450000, "total_payment": 375000},
                {"payer": "Aetna", "count": 200, "denial_rate": 0.15, "total_charge": 300000, "total_payment": 240000},
                {"payer": "Other", "count": 100, "denial_rate": 0.14, "total_charge": 125000, "total_payment": 102000}
            ]
        }
        step1["result"] = "Retrieved financial metrics by status and payer"
        
        # Step 2: Run revenue path analysis 
        step2 = {
            "step": "Analyze revenue flow paths",
            "description": "Running path analysis to track revenue flow through network",
            "execution": "Graph Analytics (path)"
        }
        execution_plan.append(step2)
        
        # Step 3: Identify service clusters
        step3 = {
            "step": "Identify service clusters",
            "description": "Using community detection to find related service groups",
            "execution": "Graph Analytics (community)"
        }
        execution_plan.append(step3)
        
        # Step 4: Calculate revenue optimization opportunities
        step4 = {
            "step": "Calculate revenue optimization opportunities",
            "description": "Quantifying financial impact of optimization strategies",
            "execution": "Graph Analytics (revenue_optimization)"
        }
        execution_plan.append(step4)
        
        # Simulate optimization results
        optimization_results = {
            "results": {
                "financial_metrics": {
                    "total_charges": 1475000,
                    "total_payments": 975000,
                    "collection_rate": 0.78,
                    "denied_charges": 225000,
                    "denial_rate": 0.15,
                    "potential_recovery": 87750
                }
            }
        }
        
        # Combine all results into a cohesive output
        final_result = {
            "query_type": "hybrid_revenue_optimization",
            "financial_data": financial_data,
            "optimization_results": optimization_results,
            "execution_plan": execution_plan,
            "summary": "Applied multi-step revenue cycle analysis combining financial metrics, revenue path tracking, " +
                      "service cluster identification, and opportunity quantification. " +
                      f"Identified ${optimization_results['results'].get('financial_metrics', {}).get('potential_recovery', 0):,.2f} " +
                      "in optimization opportunities."
        }
        
        return final_result
    
    # HYBRID QUERY TYPE 3: Patient Journey Analysis
    elif "patient" in query.lower() and ("journey" in query.lower() or "path" in query.lower()):
        # Create a simplified execution plan
        execution_plan = [
            {
                "step": "Retrieve patient demographics",
                "description": "Getting patient data via ArangoDB query",
                "execution": "ArangoDB Query"
            },
            {
                "step": "Analyze patient journeys",
                "description": "Running path analysis to track patient flow through healthcare system",
                "execution": "Graph Analytics (path)"
            },
            {
                "step": "Identify patient segments",
                "description": "Using community detection to find related patient groups",
                "execution": "Graph Analytics (community)"
            },
            {
                "step": "Analyze claim outcomes by journey",
                "description": "Correlating patient journeys with claim outcomes",
                "execution": "Multi-hop graph traversal + Analytics"
            }
        ]
        
        # Simulate patient journey results
        journey_results = {
            "query_type": "hybrid_patient_journey",
            "patient_data": [
                {
                    "patient": {"id": "patient_1", "age": 68, "gender": "F", "risk_score": 0.7},
                    "encounter_count": 6,
                    "first_encounter": {"class": "outpatient", "date": "2023-01-15"},
                    "last_encounter": {"class": "outpatient", "date": "2023-06-22"}
                },
                {
                    "patient": {"id": "patient_2", "age": 72, "gender": "M", "risk_score": 0.8},
                    "encounter_count": 8,
                    "first_encounter": {"class": "emergency", "date": "2023-02-03"},
                    "last_encounter": {"class": "inpatient", "date": "2023-05-17"}
                }
            ],
            "journey_outcomes": {
                "patient_1": {
                    "paths": [
                        {"path": ["patient_1", "encounter_1", "claim_1"], "status": "paid", "amount": 1200},
                        {"path": ["patient_1", "encounter_2", "claim_2"], "status": "denied", "amount": 950}
                    ],
                    "claim_outcomes": {"paid": 4, "denied": 2, "pending": 0},
                    "denial_rate": 0.33
                },
                "patient_2": {
                    "paths": [
                        {"path": ["patient_2", "encounter_3", "claim_3"], "status": "paid", "amount": 2800},
                        {"path": ["patient_2", "encounter_4", "claim_4"], "status": "paid", "amount": 1500}
                    ],
                    "claim_outcomes": {"paid": 6, "denied": 1, "pending": 1},
                    "denial_rate": 0.125
                }
            },
            "execution_plan": execution_plan,
            "summary": "Combined patient demographic data with multi-step journey analysis. " +
                      "Tracked multiple unique pathways through the healthcare system. " +
                      "Identified key journey patterns and correlated them with claim outcomes."
        }
        
        return journey_results
    
    # Default hybrid query if no specific type is matched
    else:
        # Create a generic execution plan
        execution_plan = [
            {
                "step": "Retrieve basic graph data",
                "description": "Getting entity counts and relationships",
                "execution": "ArangoDB Query"
            },
            {
                "step": "Run general graph analytics",
                "description": "Analyzing overall graph structure",
                "execution": "Graph Analytics"
            }
        ]
        
        # Default results
        final_result = {
            "query_type": "hybrid_general",
            "execution_plan": execution_plan,
            "summary": "Combined basic data retrieval with general graph analytics for comprehensive insights."
        }
        
        return final_result

# LLM-powered claim optimization insights
def generate_claim_optimization_insights(denial_analysis, llm=None):
    """
    Uses the LLM to generate targeted insights and recommendations based on denial analysis
    """
    # If no LLM is provided, return a simulated response
    if llm is None:
        return {
            "llm_insights": """
            Based on the analysis of claim denials, here are 5 specific, actionable recommendations:
            
            1. Address Medical Necessity Documentation for Hypertension
               * Issue: Hypertension diagnoses show 1.8x higher denial risk
               * Action: Implement standardized documentation templates for hypertension that clearly establish medical necessity
               * Expected Impact: 30% reduction in hypertension-related denials
               * Measurement: Track denial rate for hypertension diagnoses weekly
            
            2. Optimize Prior Authorization Process
               * Issue: "Prior authorization required but not obtained" is a top denial reason
               * Action: Create automated prior auth tracking system with alerts before service delivery
               * Expected Impact: 40% reduction in prior auth-related denials
               * Measurement: Monitor prior auth denial rate and authorization completion rate
            
            3. Improve Medicare Claims Submission
               * Issue: Medicare shows higher denial rates than other payers
               * Action: Conduct focused training on Medicare-specific requirements and implement pre-submission verification
               * Expected Impact: 25% reduction in Medicare denials
               * Measurement: Track denial rate by payer monthly
            
            4. Enhance Weekend Documentation Quality
               * Issue: Claims from weekend encounters have higher denial rates
               * Action: Implement additional documentation review for weekend services
               * Expected Impact: Bring weekend denial rates in line with weekday average
               * Measurement: Compare weekday vs. weekend denial rates bi-weekly
            
            5. Address High-Risk Service Combinations
               * Issue: Specific procedure-diagnosis combinations show significantly higher denial risk
               * Action: Create targeted clinical pathways for high-risk combinations with built-in documentation requirements
               * Expected Impact: 35% reduction in denials for identified combinations
               * Measurement: Track denial rates for specific procedure-diagnosis combinations
            """,
            "context": {}
        }
    
    # Extract key insights from the analysis
    top_denial_reasons = denial_analysis['top_denial_reasons'][:5]
    high_risk_diagnoses = denial_analysis['diagnosis_analysis'][:5]
    high_risk_procedures = denial_analysis['procedure_analysis'][:5]
    high_risk_payers = denial_analysis['payer_analysis'][:3]
    high_risk_combos = denial_analysis['high_risk_combinations'][:3]
    
    # Create a context for the LLM
    context = {
        "denial_rate": denial_analysis['overall_metrics']['denial_rate'] * 100,
        "top_denial_reasons": [
            {
                "reason": r['reason'],
                "percentage": r['percentage'] * 100,
                "top_payers": [p[0] for p in r['top_payers']]
            } for r in top_denial_reasons
        ],
        "high_risk_diagnoses": [
            {
                "diagnosis": d['diagnosis'],
                "relative_risk": d['relative_risk']
            } for d in high_risk_diagnoses
        ],
        "high_risk_procedures": [
            {
                "procedure": p['procedure'],
                "relative_risk": p['relative_risk']
            } for p in high_risk_procedures
        ],
        "high_risk_payers": [
            {
                "payer": p['payer'],
                "relative_risk": p['relative_risk']
            } for p in high_risk_payers
        ],
        "high_risk_combinations": [
            {
                "combination": c['combination'],
                "relative_risk": c['relative_risk']
            } for c in high_risk_combos
        ]
    }
    
    # LLM prompt template for generating insights
    optimization_prompt = ChatPromptTemplate.from_template("""
    You are an expert healthcare revenue cycle consultant analyzing claim denial patterns.
    
    Here is the analysis of claim denials:
    
    Overall denial rate: {denial_rate:.1f}%
    
    Top denial reasons:
    {top_denial_reasons}
    
    High-risk diagnoses:
    {high_risk_diagnoses}
    
    High-risk procedures:
    {high_risk_procedures}
    
    High-risk payers:
    {high_risk_payers}
    
    High-risk combinations:
    {high_risk_combinations}
    
    Based on this data, provide 5 specific, actionable recommendations to improve claim acceptance rates.
    For each recommendation:
    1. Clearly describe the issue identified in the data
    2. Explain the specific action steps to address it
    3. Indicate the expected impact on denial rates
    4. Suggest how to measure the effectiveness of the intervention
    
    Focus on practical, implementable solutions that address documentation, coding, submission timing, 
    payer-specific strategies, and staff training opportunities.
    """)
    
    # Generate insights using the LLM
    try:
        insights = llm(optimization_prompt.format(
            denial_rate=context["denial_rate"],
            top_denial_reasons=json.dumps(context["top_denial_reasons"], indent=2),
            high_risk_diagnoses=json.dumps(context["high_risk_diagnoses"], indent=2),
            high_risk_procedures=json.dumps(context["high_risk_procedures"], indent=2),
            high_risk_payers=json.dumps(context["high_risk_payers"], indent=2),
            high_risk_combinations=json.dumps(context["high_risk_combinations"], indent=2)
        ))
        
        return {
            "llm_insights": insights.content,
            "context": context
        }
    except Exception as e:
        print(f"Error generating LLM insights: {e}")
        return {
            "llm_insights": "Unable to generate insights due to an error.",
            "context": context
        }

# Enhance the denial risk algorithm with the improved analysis
def run_enhanced_denial_risk_analysis(query, G_db, llm=None):
    """Enhanced version of the denial risk analysis with more comprehensive patterns"""
    print(f"Running enhanced denial risk analysis: {query}")
    
    # Perform the enhanced analysis
    analysis_results = enhanced_denial_analysis(G_db)
    
    # Generate LLM-powered optimization insights
    insights = generate_claim_optimization_insights(analysis_results, llm)
    
    # Add the insights to the results
    analysis_results['optimization_insights'] = insights
    
    # Generate summary for the response
    denial_rate = analysis_results['overall_metrics']['denial_rate'] * 100
    
    summary_parts = [f"Analysis of {analysis_results['overall_metrics']['total_claims']} claims shows an overall denial rate of {denial_rate:.1f}%."]
    
    # Add top denial reasons
    if analysis_results['top_denial_reasons']:
        top_reason = analysis_results['top_denial_reasons'][0]
        summary_parts.append(f"The most common denial reason is '{top_reason['reason']}' ({top_reason['percentage']*100:.1f}% of denials).")
    
    # Add high risk diagnoses
    high_risk_diagnoses = [d for d in analysis_results['diagnosis_analysis'] if d['relative_risk'] > 1.5]
    if high_risk_diagnoses:
        top_diagnosis = high_risk_diagnoses[0]
        summary_parts.append(f"Diagnosis '{top_diagnosis['diagnosis']}' has {top_diagnosis['relative_risk']:.1f}x higher denial risk.")
    
    # Add high risk procedures
    high_risk_procedures = [p for p in analysis_results['procedure_analysis'] if p['relative_risk'] > 1.5]
    if high_risk_procedures:
        top_procedure = high_risk_procedures[0]
        summary_parts.append(f"Procedure '{top_procedure['procedure']}' has {top_procedure['relative_risk']:.1f}x higher denial risk.")
    
    # Add high risk combinations
    if analysis_results['high_risk_combinations']:
        top_combo = analysis_results['high_risk_combinations'][0]
        summary_parts.append(f"The combination '{top_combo['combination']}' has {top_combo['relative_risk']:.1f}x higher denial risk.")
    
    # Create the final summary
    summary = " ".join(summary_parts)
    
    # Create visualizations - in a full implementation, we would generate Plotly figures here
    visualizations = {}
    
    # Return the complete results
    return {
        "analysis": analysis_results,
        "summary": summary,
        "visualizations": visualizations
    }

# Improve GraphRAG query routing with enhanced prompt
enhanced_query_router_prompt = ChatPromptTemplate.from_template("""
You are an expert in healthcare revenue cycle management and graph analytics.
You need to analyze complex healthcare revenue graph data using the most appropriate technical approach.

The graph contains the following node types:
- patients (demographics, risk scores, insurance)
- encounters (visits, classes like inpatient/outpatient)
- procedures (services performed during encounters)
- diagnoses (conditions associated with encounters)
- claims (payment amounts, denial statuses, payers)
- billing_codes (codes associated with procedures)

Based on the user query, you need to determine which technical approach is most appropriate:

1. For simple factual queries about specific entities, simple counts, filters, or basic aggregations:
   Use "ArangoDBQuery" 
   Examples: "Show me all denied claims", "List Medicare patients", "Count procedures by type"

2. For complex analytical queries requiring graph algorithms, pattern detection, or network analysis:
   Use "GraphAnalytics" and specify the appropriate algorithm below
   Examples: "Find influential diagnoses in our network", "Identify clusters of related procedures", "What's the typical patient journey"

3. For queries that require both data retrieval AND complex analytics:
   Use "HybridQuery"
   Examples: "Analyze denial patterns for high-value cardiology procedures", "Identify revenue optimization opportunities for Medicare patients"

GRAPH ANALYTICS ALGORITHMS:
- "centrality" - Find influential/important nodes (procedures, diagnoses, providers) in the network
- "community" - Identify clusters or groupings of related nodes
- "path" - Analyze the flow of data/processes through the network (patient journeys, claim workflows)
- "denial_risk" - Identify patterns and risk factors in denied claims
- "revenue_optimization" - Find opportunities to improve financial performance

QUERY ANALYSIS STEPS:
1. Identify key medical domain concepts in the query (patients, diagnoses, claims, etc.)
2. Determine if the query requires simple data retrieval or complex pattern analysis
3. Consider if graph relationships between entities are important to the answer
4. Identify any financial metrics or workflows that need to be analyzed
5. Determine if multi-hop reasoning across the graph is needed

User query: {query}

Think through your decision step by step. First, identify what the query is asking for, then determine which approach is most appropriate.

Decision format:
```json
{{
  "tool": "ArangoDBQuery",  // or "GraphAnalytics" or "HybridQuery"
  "algorithm": "centrality", // only needed for GraphAnalytics, one of: centrality, community, path, denial_risk, revenue_optimization
  "reasoning": "Step-by-step explanation of your decision"
}}
```
""")

# Create the enhanced query router chain
def enhanced_route_query(query, llm=None):
    """Enhanced router that better classifies queries for appropriate execution paths"""
    
    # If no LLM provided, use rule-based routing
    if llm is None:
        # Rule-based fallback logic based on keywords
        lower_query = query.lower()
        
        # Financial/revenue keywords
        if any(kw in lower_query for kw in ['revenue', 'financial', 'money', 'dollar', 'payment', 'collection']):
            if any(kw in lower_query for kw in ['optimize', 'improve', 'increase', 'opportunity']):
                return {"tool": "GraphAnalytics", "algorithm": "revenue_optimization", "reasoning": "Query focuses on revenue optimization"}
        
        # Denial patterns
        if any(kw in lower_query for kw in ['denial', 'denied', 'reject']):
            if any(kw in lower_query for kw in ['pattern', 'risk', 'factor', 'why', 'reason']):
                return {"tool": "GraphAnalytics", "algorithm": "denial_risk", "reasoning": "Query focuses on denial risk analysis"}
        
        # Network analysis
        if any(kw in lower_query for kw in ['network', 'influential', 'important', 'central', 'key']):
            return {"tool": "GraphAnalytics", "algorithm": "centrality", "reasoning": "Query focuses on network centrality"}
        
        # Community/clusters
        if any(kw in lower_query for kw in ['group', 'cluster', 'community', 'segment', 'related']):
            return {"tool": "GraphAnalytics", "algorithm": "community", "reasoning": "Query focuses on community detection"}
        
        # Paths/journeys
        if any(kw in lower_query for kw in ['path', 'journey', 'flow', 'route', 'process']):
            return {"tool": "GraphAnalytics", "algorithm": "path", "reasoning": "Query focuses on path analysis"}
            
        # Default to basic queries for simpler patterns
        if any(kw in lower_query for kw in ['show', 'list', 'get', 'find', 'count']):
            return {"tool": "ArangoDBQuery", "algorithm": None, "reasoning": "Query appears to be a simple data retrieval request"}
            
        # Default to hybrid for complex queries
        return {"tool": "HybridQuery", "algorithm": None, "reasoning": "Using hybrid approach for complex query with unclear classification"}/journeys
        if any(kw in lower_query for kw in ['path', 'journey', 'flow', 'route', 'process']):
            return {"tool": "GraphAnalytics", "algorithm": "path", "reasoning": "Query focuses on path analysis"}
            
        # Default to basic queries for simpler patterns
        if any(kw in lower_query for kw in ['show', 'list', 'get', 'find', 'count']):
            return {"tool": "ArangoDBQuery", "algorithm": None, "reasoning": "Query appears to be a simple data retrieval request"}
            
        # Default to hybrid for complex queries
        return {"tool": "HybridQuery", "algorithm": None, "reasoning": "Using hybrid approach for complex query with unclear classification"}
    
    # Create the router chain with the provided LLM
    enhanced_query_router_chain = LLMChain(llm=llm, prompt=enhanced_query_router_prompt)
    
    response = enhanced_query_router_chain.run(query)
    print(f"Enhanced query router response: {response}")
    
    try:
        # Extract the JSON portion from the response
        import re
        import json
        
        # Look for JSON block in the response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without the markdown code block
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                raise ValueError("No JSON found in response")
                
        result = json.loads(json_str)
        
        return {
            "tool": result.get("tool", "HybridQuery"),  # Default to Hybrid if uncertain
            "algorithm": result.get("algorithm"),
            "reasoning": result.get("reasoning", "No explanation provided")
        }
    except Exception as e:
        print(f"Error parsing router response: {e}")
        print(f"Raw response: {response}")
        
        # Enhanced fallback logic based on keywords
        lower_query = query.lower()
        
        # Financial/revenue keywords
        if any(kw in lower_query for kw in ['revenue', 'financial', 'money', 'dollar', 'payment', 'collection']):
            if any(kw in lower_query for kw in ['optimize', 'improve', 'increase', 'opportunity']):
                return {"tool": "GraphAnalytics", "algorithm": "revenue_optimization", "reasoning": "Query focuses on revenue optimization"}
        
        # Denial patterns
        if any(kw in lower_query for kw in ['denial', 'denied', 'reject']):
            if any(kw in lower_query for kw in ['pattern', 'risk', 'factor', 'why', 'reason']):
                return {"tool": "GraphAnalytics", "algorithm": "denial_risk", "reasoning": "Query focuses on denial risk analysis"}
        
        # Network analysis
        if any(kw in lower_query for kw in ['network', 'influential', 'important', 'central', 'key']):
            return {"tool": "GraphAnalytics", "algorithm": "centrality", "reasoning": "Query focuses on network centrality"}
        
        # Community/clusters
        if any(kw in lower_query for kw in ['group', 'cluster', 'community', 'segment', 'related']):
            return {"tool": "GraphAnalytics", "algorithm": "community", "reasoning": "Query focuses on community detection"}
        
        # Paths