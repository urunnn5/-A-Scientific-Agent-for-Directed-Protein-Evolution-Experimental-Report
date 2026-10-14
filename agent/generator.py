SYSTEM_PROMPT = '''你是 GFP 定向进化科研助手。根据训练集的真实实验统计和历轮模拟反馈提出新突变。
1. 总结值得关注的位点与替换，区分单点实测证据和多点变体中的关联。
2. 生成 n 条不重复的候选，涵盖已知突变优化、单点组合与探索性替换。
   可以提出单点（1个）、双点（2个）、多点（3–5个）候选；尽量兼顾不同突变数量。
3. 每条候选相对于参考序列包含 1–5 个替换，格式如 T38S；原氨基酸必须匹配参考序列。
4. 不得编造实验值，也不得将 MLP 预测称为真实实验结果。
5. 阅读 history 中上一轮所有已评分候选及其 predicted_fitness，参考高分和低分结果修正假设。
6. 不要重复已生成过的完整突变组合；可以将历史高分单点与不同突变形成新组合。
7. 给出每个候选的 strategy、reason、risk；reason 和 risk 必须使用简体中文。
8. 只使用提供的训练集实测证据和明确标记的 MLP 预测反馈。
仅返回 JSON：{"hypotheses":[{"mutations":["T38S"],"strategy":"exploration","reason":"依据","risk":"风险"}]}'''

KNOWLEDGE_PROMPT = '''
现在额外提供氨基酸理化性质、BLOSUM62 保守替换提示，以及训练集高分单点突变知识。
请实际参考这些信息选择候选，并在 reason/risk 中说明相关性质变化、保守性或历史证据。
优先考虑有单点实验支持的替换及可能的组合，但保留部分探索性候选；组合效果尚未经实验验证。
这些性质和 BLOSUM62 只是启发式知识，不能据此直接断言荧光提升或将激进替换一律排除。
不要捏造未提供的三维结构信息、属性单位或实验测量结果。'''


GRAPH_PROMPT = '''
额外提供根据 train 构建的知识图谱查询摘要（graph）：
- 图谱只包含训练集单点变体和训练集 Fitness 前 10% 的高分变体；其余变体未进入图谱。
- 利用 Mutation→Position、Variant→Mutation、Variant→FitnessObservation 关系提出突变假设。
- 优先参考单点真实实验记录，并结合高分多点变体中的共同出现关系提出组合或新替换。
- 请在 reason 里指出所用的具体单点分数或高分组合证据，在 risk 中注明组合存在上位性风险。
- 高分多点变体中的共现突变不能直接认为单点有益；高 Fitness 也不等于已证明相比 WT 提升。
- 保持已知位点优化、组合与探索三种思路，允许 1–5 个突变，不要只推荐高分变体的复制。
- 禁止捏造图谱中不存在的关系、实验分数或 GFP 功能位点知识。
'''


class HypothesisGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, analysis, wild_type, history, n, knowledge=None, excluded=None):
        data = {
            'analysis': analysis,
            'wild_type': wild_type,
            'history': history,
            'n': n,
        }
        prompt = SYSTEM_PROMPT
        if excluded:
            data['rejected_duplicates'] = excluded
            prompt += """
            请检查 rejected_duplicates 中的突变组合。
            这些组合已经被拒绝，不得再次生成完全相同的组合。
            参考历史高分单点突变时，应设计新的组合或探索新的替换。
            允许1-5个突变，兼顾单点、双点和多点突变。
            """
        if knowledge is not None:
            data['knowledge'] = knowledge.context(analysis)
            prompt += KNOWLEDGE_PROMPT
            if hasattr(knowledge, 'graph'):
                prompt += GRAPH_PROMPT
        return self.llm.ask(prompt, data)
