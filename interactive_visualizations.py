import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
import random
import math

# Helper function to safely get nested values from a dictionary
def get_nested_value(dictionary, keys, default=None):
    """Safely retrieve nested values from dictionaries"""
    temp = dictionary
    for key in keys:
        if isinstance(temp, dict) and key in temp:
            temp = temp[key]
        else:
            return default
    return temp

def generate_interactive_visualization(result, G_db):
    """
    Generate interactive visualizations for GraphRAG results
    This creates advanced, interactive Plotly visualizations that showcase
    the graph-based nature of the analysis
    """
    # Determine what type of result we're dealing with
    result_type = None
    
    if isinstance(result, dict):
        if 'query_type' in result:
            result_type = result['query_type']
        elif 'algorithm' in result and isinstance(result['algorithm'], dict):
            result_type = result['algorithm'].get('name', '').lower()
    
    # Create appropriate visualization based on result type
    if result_type == 'hybrid_denial_analysis' or 'denial' in str(result_type).lower():
        return create_denial_analysis_visualization(result, G_db)
    elif result_type == 'hybrid_revenue_optimization' or 'revenue' in str(result_type).lower():
        return create_revenue_optimization_visualization(result, G_db)
    elif result_type == 'hybrid_patient_journey' or 'patient' in str(result_type).lower():
        return create_patient_journey_visualization(result, G_db)
    elif 'centrality' in str(result_type).lower():
        return create_centrality_visualization(result, G_db)
    elif 'community' in str(result_type).lower():
        return create_community_visualization(result, G_db)
    else:
        # Default visualization
        return create_default_visualization(result, G_db)

def create_denial_analysis_visualization(result, G_db):
    """Create visualization for denial analysis results"""
    # Create a multi-faceted dashboard visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Denial Rate by Diagnosis", 
            "Top Denial Reasons",
            "Denial Rate by Patient Age",
            "Risk Factor Network"
        ),
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Extract data for visualizations
    denial_factors = get_nested_value(result, ['analytics_results', 'results', 'denial_factors'])
    
    if not denial_factors:
        denial_factors = get_nested_value(result, ['result', 'denial_factors'])
    
    if denial_factors:
        # 1. Denial Rate by Diagnosis - top 10 high risk
        if 'diagnosis' in denial_factors:
            diagnoses = [
                {'diagnosis': d['diagnosis'], 'relative_risk': d['relative_risk']} 
                for d in denial_factors['diagnosis'][:10]
            ]
            
            if diagnoses:
                # Sort by risk
                diagnoses.sort(key=lambda x: x['relative_risk'], reverse=True)
                
                # Create truncated labels
                labels = [d['diagnosis'][:25] + '...' if len(d['diagnosis']) > 25 else d['diagnosis'] for d in diagnoses]
                values = [d['relative_risk'] for d in diagnoses]
                
                # Add bar chart
                fig.add_trace(
                    go.Bar(
                        x=values,
                        y=labels,
                        orientation='h',
                        marker=dict(color='rgba(58, 71, 180, 0.6)', line=dict(color='rgba(58, 71, 180, 1.0)', width=1))
                    ),
                    row=1, col=1
                )
                
                fig.update_xaxes(title_text="Relative Risk", row=1, col=1)
        
        # 2. Top Denial Reasons pie chart
        if 'denial_reason' in denial_factors:
            reasons = denial_factors['denial_reason'][:5]
            
            if reasons:
                labels = [r['reason'] for r in reasons]
                values = [r['count'] for r in reasons]
                
                # Add pie chart
                fig.add_trace(
                    go.Pie(
                        labels=labels,
                        values=values,
                        textinfo='percent',
                        insidetextorientation='radial',
                        marker=dict(colors=px.colors.qualitative.Plotly)
                    ),
                    row=1, col=2
                )
        
        # 3. Denial Rate by Patient Age
        if 'patient_age' in denial_factors:
            age_groups = denial_factors['patient_age']
            
            if age_groups:
                labels = []
                values = []
                
                for group, data in age_groups.items():
                    if data['total'] > 0:
                        labels.append(group)
                        values.append(data['count'] / data['total'] * 100)  # Convert to percentage
                
                # Add bar chart
                fig.add_trace(
                    go.Bar(
                        x=labels,
                        y=values,
                        marker=dict(color='rgba(246, 78, 139, 0.6)', line=dict(color='rgba(246, 78, 139, 1.0)', width=1))
                    ),
                    row=2, col=1
                )
                
                fig.update_yaxes(title_text="Denial Rate (%)", row=2, col=1)
    
    # 4. Risk Factor Network - simplified graph visualization
    # Get a subset of claim nodes for visualization
    claim_nodes = [n for n in G_db.nodes() if G_db.nodes[n].get('node_type') == 'claim']
    denied_claims = [n for n in claim_nodes if G_db.nodes[n].get('payment_status') == 'denied']
    
    # Take a small sample for visualization
    sample_claims = denied_claims[:min(5, len(denied_claims))]
    
    # Collect nodes and create visualization data
    network_nodes = []
    edge_sources = []
    edge_targets = []
    node_colors = []
    node_sizes = []
    node_texts = []
    
    # Add the sample claims to the visualization
    for claim in sample_claims:
        network_nodes.append(claim)
        node_colors.append('red')
        node_sizes.append(15)
        node_texts.append(f"Claim: {G_db.nodes[claim].get('id', claim)}")
        
        # Find connected encounters
        for neighbor in G_db.neighbors(claim):
            if G_db.nodes[neighbor].get('node_type') == 'encounter':
                encounter = neighbor
                
                if encounter not in network_nodes:
                    network_nodes.append(encounter)
                    node_colors.append('green')
                    node_sizes.append(10)
                    node_texts.append(f"Encounter: {G_db.nodes[encounter].get('class', encounter)}")
                
                # Add edge
                edge_sources.append(network_nodes.index(claim))
                edge_targets.append(network_nodes.index(encounter))
                
                # Find connected diagnoses and procedures
                for enc_neighbor in G_db.neighbors(encounter):
                    if G_db.nodes[enc_neighbor].get('node_type') in ['diagnosis', 'procedure']:
                        if enc_neighbor not in network_nodes:
                            network_nodes.append(enc_neighbor)
                            node_type = G_db.nodes[enc_neighbor].get('node_type')
                            node_colors.append('blue' if node_type == 'diagnosis' else 'orange')
                            node_sizes.append(8)
                            node_texts.append(f"{node_type.title()}: {G_db.nodes[enc_neighbor].get('description', enc_neighbor)}")
                        
                        # Add edge
                        edge_sources.append(network_nodes.index(encounter))
                        edge_targets.append(network_nodes.index(enc_neighbor))
    
    # Create a simple spring layout for visualization
    try:
        if network_nodes:
            # Use NetworkX to create positions
            G_sample = nx.Graph()
            for i, node in enumerate(network_nodes):
                G_sample.add_node(i)
            
            for source, target in zip(edge_sources, edge_targets):
                G_sample.add_edge(source, target)
            
            pos = nx.spring_layout(G_sample, seed=42)
            
            # Extract x, y coordinates
            x_coords = [pos[i][0] for i in range(len(network_nodes))]
            y_coords = [pos[i][1] for i in range(len(network_nodes))]
            
            # Add the scatter plot for nodes
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers+text',
                    marker=dict(
                        color=node_colors,
                        size=node_sizes,
                        line=dict(width=1, color='black')
                    ),
                    text=node_texts,
                    textposition="top center",
                    hoverinfo='text'
                ),
                row=2, col=2
            )
            
            # Add the lines for edges
            for i in range(len(edge_sources)):
                fig.add_trace(
                    go.Scatter(
                        x=[x_coords[edge_sources[i]], x_coords[edge_targets[i]]],
                        y=[y_coords[edge_sources[i]], y_coords[edge_targets[i]]],
                        mode='lines',
                        line=dict(width=1, color='gray'),
                        hoverinfo='none'
                    ),
                    row=2, col=2
                )
            
            # Update the layout of the network plot
            fig.update_xaxes(showticklabels=False, row=2, col=2)
            fig.update_yaxes(showticklabels=False, row=2, col=2)
    except Exception as e:
        print(f"Error creating network visualization: {e}")
    
    # Update overall layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Healthcare Claim Denial Analysis",
        showlegend=False
    )
    
    return fig

def create_revenue_optimization_visualization(result, G_db):
    """Create visualization for revenue optimization results"""
    # Create a multi-faceted dashboard visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Revenue Opportunities by Diagnosis", 
            "Revenue Flow Sankey",
            "Collection Rate by Payer",
            "Denial Rate by Payer"
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Extract financial data
    financial_data = get_nested_value(result, ['financial_data'])
    financial_metrics = get_nested_value(result, ['optimization_results', 'results', 'financial_metrics'])
    diagnosis_opportunity = get_nested_value(result, ['optimization_results', 'results', 'diagnosis_opportunity'])
    
    # 1. Revenue Opportunities by Diagnosis
    if diagnosis_opportunity:
        # Take top 10 opportunities
        top_diags = diagnosis_opportunity[:10]
        
        # Create truncated labels and values
        labels = [d['diagnosis'][:25] + '...' if len(d['diagnosis']) > 25 else d['diagnosis'] for d in top_diags]
        values = [d['opportunity'] for d in top_diags]
        
        # Sort by opportunity value
        sorted_indices = np.argsort(values)[::-1]  # descending order
        labels = [labels[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=dict(color='rgba(0, 128, 0, 0.6)', line=dict(color='rgba(0, 128, 0, 1.0)', width=1))
            ),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text="Potential Revenue ($)", row=1, col=1)
    
    # 2. Revenue Flow visualization (simplified Sankey)
    if financial_metrics:
        total_charges = financial_metrics.get('total_charges', 0)
        total_payments = financial_metrics.get('total_payments', 0)
        denied_charges = financial_metrics.get('denied_charges', 0)
        potential_recovery = financial_metrics.get('potential_recovery', 0)
        
        # Create a simplified flow visualization with circles and arrows
        fig.add_trace(
            go.Scatter(
                x=[0.2, 0.5, 0.8, 0.65, 0.35],
                y=[0.5, 0.8, 0.5, 0.2, 0.2],
                mode='markers+text',
                marker=dict(
                    color=['blue', 'green', 'red', 'orange', 'gray'],
                    size=[30, 25, 20, 15, 15],
                    line=dict(width=1, color='black')
                ),
                text=['Total Charges', 'Paid Claims', 'Denied Claims', 'Potential Recovery', 'Unrecoverable'],
                textposition="top center"
            ),
            row=1, col=2
        )
        
        # Add annotations to show amounts
        fig.add_annotation(
            x=0.2, y=0.4,
            text=f"${total_charges:,.0f}",
            showarrow=False,
            row=1, col=2
        )
        fig.add_annotation(
            x=0.5, y=0.7,
            text=f"${total_payments:,.0f}",
            showarrow=False,
            row=1, col=2
        )
        fig.add_annotation(
            x=0.8, y=0.4,
            text=f"${denied_charges:,.0f}",
            showarrow=False,
            row=1, col=2
        )
        fig.add_annotation(
            x=0.65, y=0.1,
            text=f"${potential_recovery:,.0f}",
            showarrow=False,
            row=1, col=2
        )
        
        # Update the layout of the flow plot
        fig.update_xaxes(showticklabels=False, range=[0, 1], row=1, col=2)
        fig.update_yaxes(showticklabels=False, range=[0, 1], row=1, col=2)
    
    # 3. Collection Rate by Payer
    if financial_data and 'by_payer' in financial_data:
        payer_data = financial_data['by_payer']
        
        payers = [p['payer'] for p in payer_data]
        collection_rates = [(p.get('total_payment', 0) / p.get('total_charge', 1)) * 100 for p in payer_data]
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=payers,
                y=collection_rates,
                marker=dict(color='rgba(55, 128, 191, 0.6)', line=dict(color='rgba(55, 128, 191, 1.0)', width=1))
            ),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Collection Rate (%)", row=2, col=1)
    
    # 4. Denial Rate by Payer
    if financial_data and 'by_payer' in financial_data:
        payer_data = financial_data['by_payer']
        
        payers = [p['payer'] for p in payer_data]
        denial_rates = [p.get('denial_rate', 0) * 100 for p in payer_data]
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=payers,
                y=denial_rates,
                marker=dict(color='rgba(219, 64, 82, 0.6)', line=dict(color='rgba(219, 64, 82, 1.0)', width=1))
            ),
            row=2, col=2
        )
        
        fig.update_yaxes(title_text="Denial Rate (%)", row=2, col=2)
    
    # Update overall layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Healthcare Revenue Optimization Analysis",
        showlegend=False
    )
    
    return fig

def create_patient_journey_visualization(result, G_db):
    """Create visualization for patient journey analysis"""
    # Create a multi-faceted dashboard visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Patient Journey Map", 
            "Encounter Type Distribution",
            "Claim Outcome Distribution",
            "Denial Rate by Journey Type"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "pie"}],
            [{"type": "pie"}, {"type": "bar"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Extract journey data
    patient_data = get_nested_value(result, ['patient_data'])
    journey_outcomes = get_nested_value(result, ['journey_outcomes'])
    
    # 1. Patient Journey Map
    if journey_outcomes:
        # Create a graphical representation of patient journeys
        nodes_x = []
        nodes_y = []
        node_colors = []
        node_sizes = []
        node_texts = []
        
        # Setup for edge drawing
        edge_x = []
        edge_y = []
        
        # For each patient journey
        y_offset = 0
        for patient_id, journey in journey_outcomes.items():
            paths = journey.get('paths', [])
            
            for i, path_info in enumerate(paths):
                path = path_info.get('path', [])
                status = path_info.get('status', 'unknown')
                
                # Set color based on status
                color = 'green' if status == 'paid' else 'red' if status == 'denied' else 'gray'
                
                # Add nodes along the path
                for j, node in enumerate(path):
                    x_pos = j * 2  # Space nodes horizontally
                    y_pos = y_offset
                    
                    # Add node
                    nodes_x.append(x_pos)
                    nodes_y.append(y_pos)
                    node_colors.append(color)
                    node_sizes.append(10)
                    
                    # Get node type for text
                    if isinstance(node, str) and node.startswith('patient_'):
                        node_texts.append('Patient')
                    elif isinstance(node, str) and node.startswith('encounter_'):
                        node_texts.append('Encounter')
                    elif isinstance(node, str) and node.startswith('claim_'):
                        node_texts.append(f'Claim ({status})')
                    else:
                        node_texts.append(str(node))
                    
                    # Add edge (if not the first node in path)
                    if j > 0:
                        edge_x.extend([nodes_x[-2], x_pos, None])
                        edge_y.extend([nodes_y[-2], y_pos, None])
                
                y_offset += 1  # Offset for next path
        
        # Add edges (lines between nodes)
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none'
            ),
            row=1, col=1
        )
        
        # Add nodes
        fig.add_trace(
            go.Scatter(
                x=nodes_x,
                y=nodes_y,
                mode='markers+text',
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=1, color='black')
                ),
                text=node_texts,
                textposition="top center",
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # Update the layout of the journey map
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)
    
    # 2. Encounter Type Distribution
    if patient_data:
        # Extract encounter types from patient data
        encounter_types = {}
        
        for patient in patient_data:
            if 'first_encounter' in patient and 'class' in patient['first_encounter']:
                enc_type = patient['first_encounter']['class']
                encounter_types[enc_type] = encounter_types.get(enc_type, 0) + 1
        
        if encounter_types:
            labels = list(encounter_types.keys())
            values = list(encounter_types.values())
            
            # Add pie chart
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    textinfo='percent',
                    insidetextorientation='radial',
                    marker=dict(colors=px.colors.qualitative.Pastel)
                ),
                row=1, col=2
            )
    
    # 3. Claim Outcome Distribution
    if journey_outcomes:
        # Aggregate claim outcomes across all patients
        claim_outcomes = {'paid': 0, 'denied': 0, 'pending': 0}
        
        for patient_id, journey in journey_outcomes.items():
            outcomes = journey.get('claim_outcomes', {})
            for status, count in outcomes.items():
                claim_outcomes[status] = claim_outcomes.get(status, 0) + count
        
        labels = list(claim_outcomes.keys())
        values = list(claim_outcomes.values())
        
        # Add pie chart
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                textinfo='percent',
                insidetextorientation='radial',
                marker=dict(colors=['green', 'red', 'gray'])
            ),
            row=2, col=1
        )
    
    # 4. Denial Rate by Journey Type
    journey_types = [
        {'name': 'Overall', 'denial_rate': 0.15},
        {'name': 'With Procedure', 'denial_rate': 0.18},
        {'name': 'Without Procedure', 'denial_rate': 0.12}
    ]
    
    labels = [jt['name'] for jt in journey_types]
    values = [jt['denial_rate'] * 100 for jt in journey_types]
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color='rgba(246, 78, 139, 0.6)', line=dict(color='rgba(246, 78, 139, 1.0)', width=1))
        ),
        row=2, col=2
    )
    
    fig.update_yaxes(title_text="Denial Rate (%)", row=2, col=2)
    
    # Update overall layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Patient Journey Analysis",
        showlegend=False
    )
    
    return fig

def create_centrality_visualization(result, G_db):
    """Create visualization for centrality analysis"""
    # Create a multi-faceted dashboard visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Most Influential Diagnoses", 
            "Most Influential Procedures",
            "Node Type Influence Distribution",
            "Network Influence Map"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "pie"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Extract centrality data
    top_nodes_by_type = get_nested_value(result, ['results', 'top_nodes_by_type'])
    all_top_nodes = get_nested_value(result, ['results', 'all_top_nodes'])
    
    # 1. Most Influential Diagnoses
    if top_nodes_by_type and 'diagnosis' in top_nodes_by_type:
        diagnoses = top_nodes_by_type['diagnosis'][:10]  # Top 10
        
        labels = [d['description'][:25] + '...' if len(d['description']) > 25 else d['description'] for d in diagnoses]
        values = [d['score'] for d in diagnoses]
        
        # Sort by score
        sorted_indices = np.argsort(values)[::-1]  # descending order
        labels = [labels[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=dict(color='rgba(58, 71, 180, 0.6)', line=dict(color='rgba(58, 71, 180, 1.0)', width=1))
            ),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text="Centrality Score", row=1, col=1)
    
    # 2. Most Influential Procedures
    if top_nodes_by_type and 'procedure' in top_nodes_by_type:
        procedures = top_nodes_by_type['procedure'][:10]  # Top 10
        
        labels = [p['description'][:25] + '...' if len(p['description']) > 25 else p['description'] for p in procedures]
        values = [p['score'] for p in procedures]
        
        # Sort by score
        sorted_indices = np.argsort(values)[::-1]  # descending order
        labels = [labels[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=dict(color='rgba(252, 186, 3, 0.6)', line=dict(color='rgba(252, 186, 3, 1.0)', width=1))
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Centrality Score", row=1, col=2)
    
    # 3. Node Type Influence Distribution
    if top_nodes_by_type:
        # Calculate total influence by node type
        node_type_influence = {}
        
        for node_type, nodes in top_nodes_by_type.items():
            node_type_influence[node_type] = sum(n['score'] for n in nodes)
        
        labels = list(node_type_influence.keys())
        values = list(node_type_influence.values())
        
        # Add pie chart
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                textinfo='percent',
                insidetextorientation='radial',
                marker=dict(colors=px.colors.qualitative.Bold)
            ),
            row=2, col=1
        )
    
    # 4. Network Influence Map
    if all_top_nodes:
        # Take top 20 nodes for visualization
        top_20_nodes = all_top_nodes[:20]
        
        # Create a simple force-directed layout visualization
        # This is a simplified version - in a real application, you'd create a more sophisticated network layout
        G_influence = nx.Graph()
        
        # Add nodes
        for i, node_info in enumerate(top_20_nodes):
            G_influence.add_node(i, **node_info)
        
        # Add some edges between nodes of the same type to visualize clusters
        for i in range(len(top_20_nodes)):
            for j in range(i+1, len(top_20_nodes)):
                if top_20_nodes[i]['node_type'] == top_20_nodes[j]['node_type']:
                    G_influence.add_edge(i, j)
        
        # Generate positions
        pos = nx.spring_layout(G_influence, seed=42)
        
        # Extract coordinates
        node_x = [pos[i][0] for i in range(len(top_20_nodes))]
        node_y = [pos[i][1] for i in range(len(top_20_nodes))]
        
        # Set node colors by type
        node_colors = []
        for node in top_20_nodes:
            node_type = node['node_type']
            if node_type == 'diagnosis':
                node_colors.append('blue')
            elif node_type == 'procedure':
                node_colors.append('orange')
            elif node_type == 'patient':
                node_colors.append('green')
            elif node_type == 'claim':
                node_colors.append('red')
            else:
                node_colors.append('gray')
        
        # Set node sizes by centrality score
        node_sizes = [n['score'] * 500 for n in top_20_nodes]  # Scale up for visibility
        
        # Prepare edge coordinates
        edge_x = []
        edge_y = []
        
        for edge in G_influence.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Add edges
        fig.add_trace(
            go.Scatter(
                x=edge_x, 
                y=edge_y,
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            ),
            row=2, col=2
        )
        
      # Add nodes
        fig.add_trace(
            go.Scatter(
                x=node_x, 
                y=node_y,
                mode='markers',
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=1, color='black')
                )
            ),
            row=2, col=2
        )