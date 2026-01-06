from graphviz import Digraph
import os

def create_rag_workflow():
    """
    Generates a RAG workflow diagram using Graphviz.
    Flow: Query -> Web Search -> Document Rerank -> LLM Generation -> Answer
    Includes a conditional branch for re-search.
    """
    # Create Digraph object
    dot = Digraph(name='RAG_Workflow', comment='RAG System Workflow', format='png')
    
    # Global attributes
    dot.attr(rankdir='LR') # Left to Right layout
    dot.attr('node', fontname='NanumGothic', fontsize='12')
    dot.attr('edge', fontname='NanumGothic', fontsize='10')

    # Define Nodes
    # Use box shape for standard steps
    dot.attr('node', shape='box', style='filled', fillcolor='#E1F5FE', color='#0277BD')
    dot.node('Query', 'Query\n(사용자 질문)')
    dot.node('WebSearch', 'Web Search\n(웹 검색)')
    dot.node('Rerank', 'Document Rerank\n(문서 재순위화)')
    dot.node('Generation', 'LLM Generation\n(답변 생성)')
    dot.node('Answer', 'Answer\n(최종 답변)')

    # Decision Node (Diamond shape)
    dot.attr('node', shape='diamond', style='filled', fillcolor='#FFF9C4', color='#FBC02D')
    dot.node('Check', 'Results Found?\n(검색 결과 확인)')

    # Define Edges
    dot.edge('Query', 'WebSearch')
    dot.edge('WebSearch', 'Check')
    
    # Conditional Paths
    dot.edge('Check', 'Rerank', label='Yes (있음)')
    dot.edge('Check', 'WebSearch', label='No (없음/재검색)', style='dashed', color='red')
    
    dot.edge('Rerank', 'Generation')
    dot.edge('Generation', 'Answer')

    # Output path
    output_path = os.path.join('scripts', 'rag_workflow_diagram')
    
    try:
        # Save source and render
        # Note: This requires Graphviz binary to be installed on the system.
        output_file = dot.render(filename=output_path, cleanup=True)
        print(f"Diagram successfully generated at: {output_file}")
    except Exception as e:
        print(f"Error generating diagram: {e}")
        print("Ensure Graphviz is installed and added to PATH.")
        print(f"DOT Source code:\n{dot.source}")

if __name__ == '__main__':
    create_rag_workflow()
