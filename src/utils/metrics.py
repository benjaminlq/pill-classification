from typing import List, Optional, Dict
from llama_index.schema import NodeWithScore

def calculate_precision(
    ndc_id: str, recommendation_list: List[NodeWithScore], n_recommendations: Optional[int] = None
):
    current_match = 0
    cumulative_score = 0
    for idx, node_with_score in enumerate(recommendation_list):
        if node_with_score.node.metadata["ndc11"] == ndc_id:
            current_match += 1
            cumulative_score += (current_match/(idx+1))
            if n_recommendations and (current_match >= n_recommendations):
                break
    if current_match > 0:
        return cumulative_score/current_match
    print(f"Warning: No match found for {ndc_id}")
    return 0

def mean_average_precision(
    consumer_images: List[str],
    recommendation_lists: List[List[NodeWithScore]],
    master_metadata: Dict
    ):
    map = 0
    for consumer_image, recommendation_list in zip(consumer_images, recommendation_lists):
        ndc_id = master_metadata[consumer_image]['ndc11']
        map += calculate_precision(ndc_id, recommendation_list)
    return map/len(consumer_images)

def mean_reciprocal_rank(
    consumer_images: List[str],
    recommendation_lists: List[List[NodeWithScore]],
    master_metadata: Dict
    ):
    mrr = 0
    for consumer_image, recommendation_list in zip(consumer_images, recommendation_lists):
        ndc_id = master_metadata[consumer_image]['ndc11']
        mrr += calculate_precision(ndc_id, recommendation_list, n_recommendations=1)
    return mrr/len(consumer_images)

