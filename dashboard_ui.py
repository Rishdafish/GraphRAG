import gradio as gr
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
import os

# Custom CSS for a more professional UI
custom_css = """
.gradio-container {
    background-color: #f9f9f9;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.query-history {
    max-height: 300px;
    overflow-y: auto;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 8px;
    background-color: white;
}
.visualization-container {
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: white;
    padding: 12px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.performance-metrics {
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: #f0f7ff;
    padding: 8px;
    margin-top: 8px;
}
.logo-container {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}
.logo {
    height: 80px;
}
h1 {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 20px;
}
.tabs {
    margin-top: 20px;
}
"""

# Sample queries for the UI
sample_queries = [
    "Show me all denied claims",
    "Which diagnoses have the highest denial rates?",
    "Who are the most influential patients in our network?",
    "Identify communities of related procedures",
    "What is the typical patient journey for denied claims?",
    "Show me revenue optimization opportunities for cardiac care",
    "Which payers have the highest denial rates?",
    "Identify the risk factors for claim denials",
    "What is the financial impact of documentation issues on our revenue?",
    "Show me patterns in Medicare claim denials"
]

class QueryProcessor:
    """
    Processes queries and manages history.
    In a real implementation, this would call the GraphRAG agent.
    For this demo, it creates simulated results.
    """
    def __init__(self, graphrag_agent=None, G_db=None):
        self.history = []
        self.graphrag_agent = graphrag_agent
        self.G_db = G_db
    
    def process_query(self, query):
        """Processes a new query and adds it to the history."""
        # Record query start time
        start_time = time.time()
        
        # In a real implementation, this would call the GraphRAG agent.
        if self.graphrag_agent:
            try:
                result = self.graphrag_agent(query)
            except Exception as e:
                print(f"Error calling GraphRAG agent: {e}")
                result = self.simulate_query_result(query)
        else:
            # For demo purposes, create a simulated result.
            result = self.simulate_query_result(query)
        
        # Record execution time
        execution_time = time.time() - start_time
        
        # Add to history
        self.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "result": result,
            "execution_time": execution_time
        })
        
        return result
    
    def simulate_query_result(self, query):
        """Simulates query results for demo purposes."""
        # Simulate processing time
        time.sleep(1)
        
        # Create a result based on query keywords
        if "denied" in query.lower() or "denial" in query.lower():
            return {
                "query_type": "hybrid_denial_analysis",
                "execution_plan": [
                    {"step": "Retrieve contextual data", "execution": "ArangoDB Query"},
                    {"step": "Analyze graph patterns", "execution": "Graph Analytics (CPU)"},
                    {"step": "Connect insights with multi-hop reasoning", "execution": "Graph traversal"}
                ],
                "summary": "Analysis found patterns in denied claims showing a 15.3% overall denial rate. " +
                          "Hypertension diagnoses have a 1.8x higher denial risk. " +
                          "The most common denial reason is 'Medical necessity not established' (42.3% of denials)."
            }
        elif "revenue" in query.lower() or "financial" in query.lower():
            return {
                "query_type": "hybrid_revenue_optimization",
                "execution_plan": [
                    {"step": "Retrieve financial metrics", "execution": "ArangoDB Query"},
                    {"step": "Analyze revenue flow paths", "execution": "Graph Analytics (CPU)"},
                    {"step": "Calculate optimization opportunities", "execution": "Advanced analytics"}
                ],
                "summary": "Revenue analysis identified $324,567 in potential recovery opportunities. " +
                          "The overall collection rate is 78.4% with a denial rate of 15.3%. " +
                          "Diabetes treatments present the largest opportunity at $42,890 through improved denial management."
            }
        elif "patient" in query.lower() and ("journey" in query.lower() or "path" in query.lower()):
            return {
                "query_type": "hybrid_patient_journey",
                "execution_plan": [
                    {"step": "Retrieve patient demographics", "execution": "ArangoDB Query"},
                    {"step": "Analyze patient journeys", "execution": "Graph Analytics (path)"},
                    {"step": "Identify patient segments", "execution": "Graph Analytics (community)"}
                ],
                "summary": "Combined patient demographic data with multi-step journey analysis. " +
                          "Tracked 48 unique pathways through the healthcare system. " +
                          "Identified key journey patterns and correlated them with claim outcomes."
            }
        elif "influential" in query.lower() or "important" in query.lower():
            return {
                "query_type": "centrality_analysis",
                "execution_plan": [
                    {"step": "Run PageRank", "execution": "Graph Analytics (CPU)"},
                    {"step": "Analyze influence distribution", "execution": "Network analysis"}
                ],
                "summary": "Analysis identified key influential nodes in the healthcare network. " +
                          "The most influential diagnosis is Hypertension (score: 0.0842). " +
                          "The most influential procedure is Blood Panel Test (score: 0.0673)."
            }
        else:
            return {
                "query_type": "general_analysis",
                "execution_plan": [
                    {"step": "Retrieve basic graph data", "execution": "ArangoDB Query"},
                    {"step": "Run general graph analytics", "execution": "Graph Analytics"}
                ],
                "summary": "The healthcare revenue network contains 1,284 nodes and 3,576 edges. " +
                          "It includes 100 patients, 250 encounters, 300 diagnoses, and 280 procedures. " +
                          "The overall denial rate is 15.3% with a collection rate of 78.4%."
            }
    
    def get_history(self):
        """Returns the formatted query history."""
        history_text = ""
        for i, item in enumerate(reversed(self.history)):
            history_text += f"**Query {len(self.history)-i}** ({item['timestamp']}): {item['query']}\n\n"
            history_text += f"*{item['result']['summary']}*\n\n"
            history_text += f"Execution time: {item['execution_time']:.2f}s\n\n"
            history_text += "---\n\n"
        return history_text

def create_demo_visualization(result):
    """Creates a demo visualization based on the result type."""
    from interactive_visualizations import generate_interactive_visualization
    import networkx as nx
    
    # Create a small demo graph for visualization
    G_demo = nx.Graph()
    
    # Add some sample nodes of different types
    for i in range(5):
        G_demo.add_node(f"patient_{i}", node_type="patient", id=f"P{i}", gender="F" if i % 2 else "M", age=60+i)
        G_demo.add_node(f"encounter_{i}", node_type="encounter", id=f"E{i}", encounter_class="outpatient")
        G_demo.add_node(f"diagnosis_{i}", node_type="diagnosis", id=f"D{i}", description=f"Diagnosis {i}")
        G_demo.add_node(f"procedure_{i}", node_type="procedure", id=f"PR{i}", description=f"Procedure {i}")
        G_demo.add_node(f"claim_{i}", node_type="claim", id=f"C{i}", payment_status="denied" if i % 3 == 0 else "paid", 
                        total_charge=1000+i*100, payment_amount=800+i*80 if i % 3 != 0 else 0,
                        denial_reason="Medical necessity" if i % 3 == 0 else None)
    
    # Add edges to connect the nodes
    for i in range(5):
        G_demo.add_edge(f"patient_{i}", f"encounter_{i}", edge_type="patient_encounter")
        G_demo.add_edge(f"encounter_{i}", f"diagnosis_{i}", edge_type="encounter_diagnosis")
        G_demo.add_edge(f"encounter_{i}", f"procedure_{i}", edge_type="encounter_procedure")
        G_demo.add_edge(f"encounter_{i}", f"claim_{i}", edge_type="encounter_claim")
    
    # Use the actual visualization function
    try:
        from interactive_visualizations import generate_interactive_visualization
        return generate_interactive_visualization(result, G_demo)
    except Exception as e:
        print(f"Error generating visualization: {e}")
        # Fallback to a default Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3]))
        fig.update_layout(title="Demo Visualization")
        return fig

def process_and_visualize(query, processor):
    """Processes the query and generates a visualization."""
    if not query:
        return None, "Please enter a query to analyze healthcare revenue data."
    
    # Process the query
    result = processor.process_query(query)
    
    # Generate plot
    fig = create_demo_visualization(result)
    
    # Generate response text
    response = f"## Query Results\n\n"
    response += f"{result['summary']}\n\n"
    response += f"### Execution Plan\n\n"
    
    for step in result['execution_plan']:
        response += f"- **{step['step']}**: {step['execution']}\n"
    
    return fig, response

def launch_ui(graphrag_agent=None, G_db=None):
    """Launches the UI for the Healthcare GraphRAG system."""
    # Initialize the processor
    processor = QueryProcessor(graphrag_agent, G_db)
    
    # Define the interface
    with gr.Blocks(css=custom_css) as demo:
        # Header and description
        gr.HTML("""
        <div class="logo-container">
            <img src="https://img.icons8.com/color/96/000000/network.png" alt="Graph Network" class="logo">
        </div>
        <h1>Healthcare Revenue Cycle GraphRAG</h1>
        """)
        
        gr.Markdown("""
        This application uses **GraphRAG** (Graph-based Retrieval Augmented Generation) to analyze healthcare 
        revenue cycle data. Ask questions about claims, denials, patient journeys, and revenue optimization.
        
        The system dynamically selects between ArangoDB graph queries, graph analytics, or hybrid approaches.
        """)
        
        # Input components
        with gr.Row():
            query_input = gr.Textbox(
                label="Ask a question about your healthcare revenue data",
                placeholder="e.g., Which diagnoses have the highest denial rates?",
                lines=2
            )
        
        # Sample queries
        gr.Examples(
            examples=sample_queries,
            inputs=query_input,
            label="Sample Questions"
        )
        
        # Submit button
        submit_btn = gr.Button("Analyze with GraphRAG", variant="primary")
        
        # Output tabs
        with gr.Tabs() as tabs:
            with gr.TabItem("Analysis Results"):
                with gr.Row():
                    with gr.Column(scale=3):
                        plot_output = gr.Plot(label="Interactive Visualization")
                    with gr.Column(scale=2):
                        result_output = gr.Markdown(label="Analysis")
            
            with gr.TabItem("Query History"):
                history_output = gr.Markdown()
                
                # Add a refresh button for the history
                refresh_btn = gr.Button("Refresh History")
                
                # Update the history when the refresh button is clicked
                refresh_btn.click(
                    fn=lambda: processor.get_history(),
                    outputs=history_output
                )
            
            with gr.TabItem("About GraphRAG"):
                gr.Markdown("""
                ## About Healthcare Revenue Cycle GraphRAG
                
                This application showcases the power of **Graph-based Retrieval Augmented Generation (GraphRAG)** for healthcare revenue cycle analysis.
                
                ### Key Features:
                
                - **Relationship-Aware Analysis**: Unlike traditional vector-based RAG, GraphRAG preserves and leverages relationships between entities (patients, claims, procedures, etc.).
                
                - **Graph Analytics**: Utilizes algorithms like PageRank, community detection, and path analysis.
                
                - **Hybrid Query Execution**: Dynamically routes queries between ArangoDB graph traversal and graph analytics based on the query intent.
                
                - **Multi-Hop Reasoning**: Connects information across multiple hops in the graph to generate insights that would be impossible with traditional approaches.
                
                - **Advanced Visualization**: Interactive visualizations that highlight the graph structure and analytics results.
                
                ### Architecture:
                
                1. **Query Analysis**: Natural language queries are analyzed to determine the optimal execution strategy.
                
                2. **Execution Engine**: Queries are routed to either ArangoDB (for specific data retrieval), graph analytics (for pattern discovery), or a hybrid approach.
                
                3. **Result Synthesis**: Results are combined with the graph context to generate comprehensive natural language responses.
                
                4. **Visualization**: Interactive visualizations are generated to help interpret the results.
                """)
        
        # Connect the button to the processing function
        submit_btn.click(
            fn=lambda q: process_and_visualize(q, processor),
            inputs=query_input,
            outputs=[plot_output, result_output]
        )
        
        # Initialize the history tab
        demo.load(
            fn=lambda: processor.get_history(),
            outputs=history_output
        )
    
    # Launch the app
    demo.launch(share=False)

if __name__ == "__main__":
    launch_ui()