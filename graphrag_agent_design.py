from langchain.agents import Tool, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, TypedDict, Annotated
import inspect
import json

# Define state for the agent
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    current_query: str
    execution_plan: Dict[str, Any]
    intermediate_steps: List[Dict[str, Any]]
    results: Dict[str, Any]
    final_answer: str

# Agent system prompt that showcases the structured reasoning GraphRAG enables
HEALTHCARE_GRAPHRAG_PROMPT = """
You are HealthRevAI, an intelligent GraphRAG agent specialized in healthcare revenue cycle analysis.

You have these capabilities:
1. AQL Database Queries - For retrieving specific facts from the graph database
2. GPU-Accelerated Graph Analytics - For complex pattern analysis (though we'll use CPU fallback when on Mac)
3. Hybrid Query Execution - For combining both approaches with structured reasoning

The healthcare revenue graph contains:
- Patients (demographics, risk scores)
- Encounters (visits, dates, providers)
- Diagnoses (ICD codes, descriptions)
- Procedures (CPT codes, descriptions)
- Claims (charges, payments, denials)
- Billing codes (fee schedules)

All these entities are connected in a graph structure that allows multi-hop traversal and 
relationship-aware analytics.

For each user query:
1. First, analyze the query type to determine optimal execution strategy
2. For factual queries about specific entities, use AQL
3. For pattern discovery or influence analysis, use Graph Analytics
4. For complex questions requiring both context and analytics, use Hybrid Execution
5. Always explain your reasoning process, showing how the graph structure enabled insights
   that would be impossible with traditional vector-based retrieval

In your responses, highlight how graph relationships enabled the answer and visualize the 
relevant subgraphs when helpful.
"""

# Create structured GraphRAG agent with explicit graph reasoning stages
def create_graphrag_healthcare_agent(
    llm, 
    arangodb_query_tool, 
    graph_analytics_tool, 
    hybrid_query_tool,
    visualization_generator
):
    """
    Create a GraphRAG agent for healthcare revenue cycle analysis
    
    Args:
        llm: Language model to use for reasoning
        arangodb_query_tool: Function to execute ArangoDB queries
        graph_analytics_tool: Function to execute graph analytics
        hybrid_query_tool: Function to execute hybrid queries
        visualization_generator: Function to generate visualizations
        
    Returns:
        A function that takes a query and returns a response
    """
    # Create tools with specific purposes
    tools = [
        Tool(
            name="ArangoDBQuery",
            func=arangodb_query_tool,
            description="Retrieve specific factual data about patients, claims, or procedures using ArangoDB graph traversal"
        ),
        Tool(
            name="GraphAnalytics",
            func=lambda q: graph_analytics_tool(q),
            description="Execute complex graph algorithms like centrality, community detection, or path analysis"
        ),
        Tool(
            name="HybridQuery",
            func=hybrid_query_tool,
            description="Combine graph database queries and analytics for complex, multi-step reasoning that uses both graph structure and patterns"
        ),
        Tool(
            name="GenerateVisualization",
            func=visualization_generator,
            description="Create an interactive visualization of the relevant graph subgraph, algorithm results, or data insights"
        )
    ]
    
    # Create the prompt with explicit reasoning steps
    prompt = ChatPromptTemplate.from_messages([
        ("system", HEALTHCARE_GRAPHRAG_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("system", """
        Approach this query using structured graph reasoning:
        
        1. ANALYZE: What is the user asking for? What graph entities and relationships are relevant?
        2. PLAN: What query approach is best - AQL, Graph Analytics, or Hybrid? What specific algorithms or traversals do I need?
        3. EXECUTE: Run the appropriate tools with optimized parameters
        4. SYNTHESIZE: Connect the results using the graph structure to provide a comprehensive answer
        5. VISUALIZE: Generate an appropriate graph visualization that shows the key relationships
        
        Always walk through each step explicitly to showcase the power of graph-based reasoning.
        """),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # For Mac implementation without LangGraph, we'll use a simpler approach
    # that doesn't require the full graph-based agent workflow
    
    def graphrag_agent(query):
        """Execute the GraphRAG agent on a query"""
        from enhanced_hybrid_query import enhanced_route_query
        
        print(f"Processing query: {query}")
        
        # Step 1: Route the query to determine the execution path
        route = enhanced_route_query(query, llm)
        print(f"Query routed to: {route['tool']}")
        print(f"Reasoning: {route['reasoning']}")
        
        # Step 2: Execute the query using the selected tool
        if route["tool"] == "ArangoDBQuery":
            # Simple data retrieval with ArangoDB
            result = arangodb_query_tool(query)
            execution_path = "ArangoDB Query"
            
        elif route["tool"] == "GraphAnalytics":
            # Analytics with NetworkX
            result = graph_analytics_tool(query, algorithm=route.get("algorithm"))
            execution_path = f"Graph Analytics ({route.get('algorithm', 'general')})"
            
        else:  # HybridQuery
            # Combined approach for complex questions
            result = hybrid_query_tool(query)
            execution_path = "Hybrid Query (ArangoDB + Graph Analytics)"
        
        # Step 3: Generate visualization if needed
        try:
            visualization = visualization_generator(result)
            result["visualization"] = visualization
        except Exception as e:
            print(f"Error generating visualization: {e}")
        
        # Step 4: Generate natural language response
        # In a full implementation, this would be handled by the LLM
        # For this simplified version, we'll extract the summary from the result
        if isinstance(result, dict) and "summary" in result:
            response = result["summary"]
        else:
            response = "Analysis complete. Please see the visualization for detailed results."
        
        return response
    
    return graphrag_agent

# Function to set up the GraphRAG agent
def setup_healthcare_graphrag_agent(llm, G_db):
    """
    Set up the healthcare GraphRAG agent with all necessary tools
    
    Args:
        llm: Language model to use for reasoning
        G_db: NetworkX graph from ArangoDB
        
    Returns:
        A function that takes a query and returns a response
    """
    from gpu_accelerated_analytics import run_enhanced_graph_analytics
    from enhanced_hybrid_query import run_enhanced_hybrid_query
    from interactive_visualizations import generate_interactive_visualization
    
    # Function to run ArangoDB query
    def run_arangodb_query(query):
        """Execute an ArangoDB query"""
        print(f"Running ArangoDB query: {query}")
        
        # In a real implementation, this would convert the query to AQL and execute it
        # For this demo, we'll return a simple mock result
        return {
            "success": True,
            "results": [
                {"claim_id": "CL1", "payment_status": "denied", "total_charge": 1200, "denial_reason": "Medical necessity not established"},
                {"claim_id": "CL2", "payment_status": "paid", "total_charge": 850, "payment_amount": 680},
                {"claim_id": "CL3", "payment_status": "denied", "total_charge": 1500, "denial_reason": "Missing documentation"}
            ],
            "query_type": "arangodb"
        }
    
    # Create the agent
    agent = create_graphrag_healthcare_agent(
        llm=llm,
        arangodb_query_tool=run_arangodb_query,
        graph_analytics_tool=lambda query, algorithm=None: run_enhanced_graph_analytics(query, G_db, algorithm),
        hybrid_query_tool=lambda query: run_enhanced_hybrid_query(query, G_db, llm),
        visualization_generator=lambda result: generate_interactive_visualization(result, G_db)
    )
    
    return agent