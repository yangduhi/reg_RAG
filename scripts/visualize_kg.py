from pyvis.network import Network
import os

def visualize_knowledge_graph(nodes_data, edges_data):
    # 1. Network 객체 생성 (계층 구조 레이아웃 적용)
    net = Network(
        height='600px', 
        width='100%', 
        bgcolor='#ffffff', 
        font_color='black', 
        notebook=False,
        directed=True,
        layout=True # 계층 구조 활성화
    )

    # 2. 그룹별 색상 및 모양 정의
    # 입력/출력: 원형, 처리: 박스, 데이터: 데이터베이스 모양(아이콘 대체)
    group_options = {
        'Input': {'color': '#FFCDD2', 'shape': 'ellipse'},      # 빨강 (사용자)
        'Process': {'color': '#BBDEFB', 'shape': 'box'},        # 파랑 (처리)
        'Search': {'color': '#FFF9C4', 'shape': 'box'},         # 노랑 (검색)
        'Data': {'color': '#E1BEE7', 'shape': 'database'},      # 보라 (데이터)
        'Output': {'color': '#C8E6C9', 'shape': 'star'}         # 초록 (결과)
    }

    # 3. 노드 추가
    for node in nodes_data:
        grp = node['group']
        opts = group_options.get(grp, {})
        
        tooltip_info = (
            f"<div style='padding:5px;'>"
            f"<h4>{node['label']}</h4>"
            f"<p>{node.get('info', '')}</p>"
            f"</div>"
        )
        
        net.add_node(
            node['id'], 
            label=node['label'], 
            title=tooltip_info,
            group=grp,
            color=opts.get('color', '#eee'),
            shape=opts.get('shape', 'dot'),
            level=node.get('level', 0), # 계층 레벨
            font={'face': 'Malgun Gothic', 'size': 16} # 한글 폰트
        )

    # 4. 엣지 추가
    for edge in edges_data:
        net.add_edge(
            edge['from'], 
            edge['to'], 
            label=edge.get('label', ''),
            arrows='to',
            width=2,
            color='#999'
        )

    # 5. 물리 엔진 및 레이아웃 설정 (계층형)
    net.set_options("""
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 200,
          "nodeSpacing": 150
        }
      },
      "physics": {
        "hierarchicalRepulsion": {
          "centralGravity": 0,
          "springLength": 100,
          "springConstant": 0.01,
          "nodeDistance": 120,
          "damping": 0.09
        },
        "minVelocity": 0.75,
        "solver": "hierarchicalRepulsion"
      }
    }
    """)

    # 6. 파일 저장
    output_file = 'graph.html'
    net.save_graph(output_file)
    print(f"성공: 워크플로우 그래프가 '{os.path.abspath(output_file)}'로 생성되었습니다.")

if __name__ == "__main__":
    # RAG 워크플로우 데이터 (한글)
    nodes = [
        # Level 0: 입력
        {'id': 'User', 'label': '사용자 질문', 'group': 'Input', 'level': 0, 'info': 'Streamlit을 통해 입력된 자연어 질문'},
        
        # Level 1: 변환
        {'id': 'Transform', 'label': '질의 변환\n(LLM)', 'group': 'Process', 'level': 1, 'info': '질문을 분석하여 최적화된 검색어 생성'},
        
        # Level 2: 쿼리 확장
        {'id': 'Q_Ko', 'label': '한국어 쿼리\n(별표/시험방법)', 'group': 'Search', 'level': 2, 'info': 'KMVSS 상세 기준 검색용'},
        {'id': 'Q_En', 'label': '영어 쿼리\n(ECE/FMVSS)', 'group': 'Search', 'level': 2, 'info': '국제 규정 검색용'},
        
        # Level 3: 검색
        {'id': 'Retrieve', 'label': '하이브리드 검색\n(BM25+Vector)', 'group': 'Process', 'level': 3, 'info': 'ChromaDB에서 Top-K 문서 추출'},
        
        # Level 4: 재순위화 (분기)
        {'id': 'Rerank_Ko', 'label': '한국어 문서\n(순위 유지)', 'group': 'Process', 'level': 4, 'info': '검색 점수 신뢰 (Reranker 왜곡 방지)'},
        {'id': 'Rerank_En', 'label': '영어 문서\n(재순위화)', 'group': 'Process', 'level': 4, 'info': 'FlashRank(ms-marco) 적용'},
        
        # Level 5: 병합 및 평가
        {'id': 'Interleave', 'label': '교차 병합\n(Interleaving)', 'group': 'Process', 'level': 5, 'info': '1:1 비율로 문서 섞기'},
        {'id': 'Grade', 'label': '적합성 평가\n(LLM Grader)', 'group': 'Process', 'level': 6, 'info': '질문과 문서의 관련성 검증'},
        
        # Level 6: 생성
        {'id': 'Context', 'label': '문맥(Context)\n구성', 'group': 'Data', 'level': 7, 'info': '검증된 문서만 LLM에 주입'},
        {'id': 'Generate', 'label': '답변 생성\n(Gemini 2.0)', 'group': 'Process', 'level': 8, 'info': '최종 답변 및 출처 작성'},
        
        # Level 7: 출력
        {'id': 'Answer', 'label': '최종 답변', 'group': 'Output', 'level': 9, 'info': '사용자에게 제공되는 결과'}
    ]

    edges = [
        {'from': 'User', 'to': 'Transform'},
        {'from': 'Transform', 'to': 'Q_Ko', 'label': '확장'},
        {'from': 'Transform', 'to': 'Q_En', 'label': '번역'},
        {'from': 'Q_Ko', 'to': 'Retrieve'},
        {'from': 'Q_En', 'to': 'Retrieve'},
        {'from': 'Retrieve', 'to': 'Rerank_Ko', 'label': 'KR Docs'},
        {'from': 'Retrieve', 'to': 'Rerank_En', 'label': 'EN Docs'},
        {'from': 'Rerank_Ko', 'to': 'Interleave'},
        {'from': 'Rerank_En', 'to': 'Interleave'},
        {'from': 'Interleave', 'to': 'Grade'},
        {'from': 'Grade', 'to': 'Context', 'label': 'Pass'},
        {'from': 'Context', 'to': 'Generate'},
        {'from': 'Generate', 'to': 'Answer'}
    ]

    visualize_knowledge_graph(nodes, edges)
