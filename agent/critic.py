REVIEW_PROMPT = '''你是 GFP 定向进化科研评审。基于训练集实验统计和 MLP 预测评价本轮 Top-k 推荐。
1. 为每条推荐给出 assessment、critic_risk，并指出实验支持与不确定性。
2. 分析较低分候选的可能问题，提出下一轮探索建议。
3. 不允许改写分数与排名。
assessment 和 critic_risk 使用简体中文；下一轮建议也使用简体中文。
仅返回 JSON：{"reviews":[{"mutant":"T38S","assessment":"评价","critic_risk":"风险"}],"next_round_suggestions":["建议"]}'''

KNOWLEDGE_REVIEW = '''
如果提供 knowledge_evidence，应结合 BLOSUM62 和理化性质变化评价候选，并解释历史支持与未知风险。
不可仅依据单项性质认定 Fitness 一定升高或降低。'''


GRAPH_REVIEW = '''
如果有图谱证据：分别写出单点实测支持、高分多点共现支持及无证据部分。
不能由高分共现直接断言突变本身提高 Fitness，也不能把 MLP 预测称为真实实验。
允许讨论非加性突变效应；不捏造结构/功能信息。
'''


class ScientificCritic:
    def __init__(self, llm):
        self.llm = llm

    def review(self, evaluated, analysis, knowledge=None):
        top = [{k: item[k] for k in ('mutant', 'rank', 'predicted_fitness', 'reason', 'risk')}
               for item in evaluated['top_k']]
        low = [{'mutant': c['mutant'], 'predicted_fitness': c['predicted_fitness']}
               for c in evaluated['all_scored'][-3:]]
        data = {'top_candidates': top, 'low_score_examples': low, 'analysis': analysis}
        prompt = REVIEW_PROMPT
        if knowledge is not None:
            data['knowledge_evidence'] = {
                item['mutant']: knowledge.evidence(item['mutant']) for item in top
            }
            prompt += KNOWLEDGE_REVIEW
            if hasattr(knowledge, 'graph'):
                prompt += GRAPH_REVIEW
        result = self.llm.ask(prompt, data)
        reviews = {item['mutant']: item for item in result['reviews']}
        recommendations = [
            {**item,
             'assessment': reviews.get(item['mutant'], {}).get('assessment', ''),
             'critic_risk': reviews.get(item['mutant'], {}).get('critic_risk', '')}
            for item in evaluated['top_k']
        ]
        return recommendations, result['next_round_suggestions']
