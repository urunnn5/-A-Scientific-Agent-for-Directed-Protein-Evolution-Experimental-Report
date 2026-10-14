import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent.analyst import DataAnalyst
from agent.critic import ScientificCritic
from agent.designer import MutationDesigner, infer_wild_type
from agent.evaluator import FitnessEvaluator
from agent.generator import HypothesisGenerator
from agent.llm import DeepSeekLLM
from agent.knowledge import KnowledgeBase
from agent.graph import GraphKnowledge
from agent.predictor import MLPPredictor
from fitness.train import MODEL_PATH

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results'


def compare_baselines(predictor, test, rounds, budget, top_k, seed=42):
    # 相同隐藏测试集候选池：随机选择 vs MLP 直接排序。
    rng = np.random.default_rng(seed)
    pools = rng.permutation(len(test))[:rounds * budget].reshape(rounds, budget)
    rows = []
    best = {'random': float('-inf'), 'fitness_only': float('-inf')}
    for round_id, indices in enumerate(pools, 1):
        pool = test.iloc[indices].reset_index(drop=True)
        scores = predictor.predict(pool['mutated_sequence'].tolist())
        choices = {
            'random': rng.choice(budget, top_k, replace=False),
            'fitness_only': np.argsort(scores)[-top_k:],
        }
        for method, chosen in choices.items():
            measured = pool['DMS_score'].iloc[chosen]
            best[method] = max(best[method], float(measured.max()))
            rows.append({
                'method': method,
                'round': round_id,
                'top_k_mean': float(measured.mean()),
                'round_best': float(measured.max()),
                'best_so_far': best[method],
                'score_source': 'held_out_experiment',
            })
    return pd.DataFrame(rows)

def run_evolution(train, predictor, llm, rounds, budget, top_k, knowledge=None):
    wild_type = infer_wild_type(train)
    analysis = DataAnalyst().analyze(train)
    generate = HypothesisGenerator(llm)
    design = MutationDesigner()
    evaluate = FitnessEvaluator(predictor)
    critic = ScientificCritic(llm)
    known = set(train['mutated_sequence'])
    history, summaries, all_rows = [], [], []
    best_so_far = None

    for round_id in range(1, rounds + 1):
        candidates, rejected = [], []
        generated = 0

        # 获取之前轮次因重复被拒绝的突变
        excluded = [
            name for record in history
            for name in record.get('rejected_duplicates', [])
        ][-60:]

        # 每轮最多尝试4次，累计有效候选
        for attempt in range(4):
            remaining = budget - len(candidates)

            if remaining <= 0:
                break

            report = generate.generate(analysis, wild_type, history, remaining, knowledge, excluded=excluded)
            hypotheses = report['hypotheses'][:remaining]
            generated += len(hypotheses)

            # 排除训练集、历史轮次和本轮已有序列
            seen = known | {item['mutated_sequence'] for item in candidates}

            new_candidates = design.design(hypotheses, wild_type, seen)
            candidates.extend(new_candidates)
            rejected.extend(design.rejections)

            # 记录重复候选，反馈给下一次LLM生成
            duplicates = [
                ':'.join(item['mutations'])
                for item in design.rejections
                if item['reason'] == 'duplicate'
            ]

            excluded = list(dict.fromkeys(excluded + duplicates))[-60:]

            if len(candidates) < budget and attempt < 3:
                print(f'第 {round_id} 轮第 {attempt + 1} 次生成：有效 {len(candidates)}/{budget}，继续补充')

        # 所有有效候选统一进行MLP预测
        evaluated = evaluate.evaluate(candidates, top_k)
        known.update(item['mutated_sequence'] for item in candidates)

        if evaluated['top_k']:
            recommendations, suggestions = critic.review(evaluated, analysis, knowledge)
            round_best = evaluated['top_k'][0]['predicted_fitness']
            best_so_far = max(best_so_far, round_best) if best_so_far is not None else round_best
        else:
            recommendations, suggestions, round_best = [], ['探索新的突变位点'], None

        # 保存所有有效候选
        reviews = {item['mutant']: item for item in recommendations}

        for item in evaluated['all_scored']:
            review = reviews.get(item['mutant'], {})
            all_rows.append({
                'round': round_id,
                'status': 'scored',
                'reject_reason': '',
                'score_source': 'mlp_predicted',
                'in_top_k': item['rank'] <= top_k,
                **item,
                'assessment': review.get('assessment', ''),
                'critic_risk': review.get('critic_risk', ''),
            })

        # 保存所有被拒绝的候选
        for item in rejected:
            all_rows.append({
                'round': round_id,
                'status': 'rejected',
                'mutant': ':'.join(item['mutations']),
                'mutation_count': len(item['mutations']),
                'reject_reason': item['reason'],
                'in_top_k': False,
            })

        # 每轮结果汇总
        summaries.append({
            'round': round_id,
            'generated': generated,
            'valid': len(candidates),
            'rejected': len(rejected),
            'round_best': round_best,
            'best_so_far': best_so_far,
            'score_source': 'mlp_predicted',
            'next_round_suggestions': '; '.join(suggestions),
        })

        # 提取低分候选作为下一轮反馈
        low_cases = [
            {
                'mutant': item['mutant'],
                'predicted_fitness': item['predicted_fitness'],
                'reason': item['risk']
            }
            for item in evaluated['all_scored'][-3:]
        ]

        # 保存本轮全部已评分候选和反馈
        history.append({
            'round': round_id,
            'top_k': [
                {'mutant': c['mutant'], 'predicted_fitness': c['predicted_fitness']}
                for c in recommendations
            ],
            'scored_candidates': [
                {
                    'mutant': c['mutant'],
                    'mutation_count': c['mutation_count'],
                    'predicted_fitness': c['predicted_fitness'],
                    'strategy': c['strategy']
                }
                for c in evaluated['all_scored']
            ],
            'low_score_cases': low_cases,
            'next_round_suggestions': suggestions,
            'rejected_duplicates': excluded,
        })

        print(f'第 {round_id} 轮：有效候选 {len(candidates)}/{budget}，最高预测 Fitness {round_best}')

    return pd.DataFrame(summaries), pd.DataFrame(all_rows)



def plot_curve(data, output):
    fig, ax = plt.subplots()
    for method, part in data.groupby('method'):
        ax.plot(part['round'], part['best_so_far'], marker='o', label=method)
    ax.set_xlabel('Round')
    ax.set_ylabel('Best fitness so far')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='GFP directed evolution')
    parser.add_argument('--mode', choices=['baseline', 'agent'], default='baseline')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--n', type=int, default=30, help='每轮候选数')
    parser.add_argument('--top-k', type=int, default=5)
    enhancement = parser.add_mutually_exclusive_group()
    enhancement.add_argument('--knowledge', action='store_true', help='通用氨基酸知识增强')
    enhancement.add_argument('--graph', action='store_true', help='知识图谱增强（包含通用氨基酸知识）')
    args = parser.parse_args()

    predictor = MLPPredictor.load(MODEL_PATH)
    OUTPUT.mkdir(exist_ok=True)

    if args.mode == 'baseline':
        if args.knowledge or args.graph:
            parser.error('--knowledge / --graph 仅用于 --mode agent')
        from fitness.train import report
        model, encoder = predictor.model, predictor.encoder
        report(model, encoder)
        test = pd.read_csv(ROOT / 'data' / 'test.csv')
        scores = compare_baselines(predictor, test, args.rounds, args.n, args.top_k)
        scores.to_csv(OUTPUT / 'baselines.csv', index=False)
        plot_curve(scores, OUTPUT / 'fitness_curve.png')
        print(scores.to_string(index=False))
        return

    train = pd.read_csv(ROOT / 'data' / 'train.csv')
    if args.graph:
        knowledge = GraphKnowledge(ROOT / 'data' / 'aa_props.csv', train)
        print('训练集知识图谱：', knowledge.stats)
    elif args.knowledge:
        knowledge = KnowledgeBase(ROOT / 'data' / 'aa_props.csv')
    else:
        knowledge = None
    summaries, candidates = run_evolution(
        train, predictor, DeepSeekLLM(), args.rounds, args.n, args.top_k, knowledge
    )
    output = OUTPUT / ('agent_graph' if args.graph else 'agent_knowledge' if args.knowledge else 'agent_plain')
    output.mkdir(exist_ok=True)
    summaries.to_csv(output / 'rounds.csv', index=False, encoding='utf-8-sig')
    columns = ['round', 'status', 'mutant', 'mutation_count', 'strategy',
               'predicted_fitness', 'rank', 'in_top_k', 'score_source',
               'reason', 'risk', 'assessment', 'critic_risk',
               'reject_reason', 'mutated_sequence']
    candidates.reindex(columns=columns).to_csv(
        output / 'candidates.csv', index=False, encoding='utf-8-sig'
    )
    for name in ['agent_rounds.csv', 'all_scored.csv', 'top_k.csv',
                 'agent_details.json', 'agent_curve.png']:
        (output / name).unlink(missing_ok=True)
    for file in output.glob('round_*.csv'):
        file.unlink()
    print(f'结果已保存：{output / "rounds.csv"}、{output / "candidates.csv"}')



if __name__ == '__main__':
    main()
