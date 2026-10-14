"""从训练集的单点变体和前 10% 高分变体构建小型实验知识图谱。"""
from collections import Counter

import networkx as nx

from agent.knowledge import KnowledgeBase


class GraphKnowledge(KnowledgeBase):
    """NetworkX 属性图；只记录训练集实测证据，不把高分误称为因果改善。"""

    def __init__(self, properties_path, train):
        super().__init__(properties_path)
        self.threshold = float(train['DMS_score'].quantile(0.9))
        singles = train['mutant'].str.count(':').eq(0)
        highs = train['DMS_score'].ge(self.threshold)
        selected = train.loc[singles | highs]
        self.graph = nx.MultiDiGraph()
        self._build(selected)
        self.stats = {
            'threshold': round(self.threshold, 5),
            'train_samples': len(train),
            'selected_variants': len(selected),
            'single_variants': int(singles.sum()),
            'high_variants': int(highs.sum()),
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
        }

    def _build(self, selected):
        graph = self.graph
        for aa, props in self.properties.items():
            aa_id = f'aa:{aa}'
            graph.add_node(aa_id, type='AminoAcid', symbol=aa)
            for name, value in props.items():
                prop_id = f'property:{name}'
                graph.add_node(prop_id, type='Property', name=name)
                graph.add_edge(aa_id, prop_id, relation='has_property', value=float(value))

        for row in selected.itertuples(index=False):
            variant_id = f'variant:{row.mutant}'
            obs_id = f'fitness:{row.mutant}'
            mutations = row.mutant.split(':')
            is_high = float(row.DMS_score) >= self.threshold
            graph.add_node(variant_id, type='Variant', mutant=row.mutant,
                           mutation_count=len(mutations))
            graph.add_node(obs_id, type='FitnessObservation',
                           score=float(row.DMS_score), high_top10=bool(is_high),
                           source='measured_train')
            graph.add_edge(variant_id, obs_id, relation='has_fitness')
            for name in mutations:
                mut_id = f'mutation:{name}'
                position = int(name[1:-1])
                graph.add_node(mut_id, type='Mutation', name=name)
                graph.add_node(f'position:{position}', type='Position', index=position)
                graph.add_edge(variant_id, mut_id, relation='contains')
                graph.add_edge(mut_id, f'position:{position}', relation='occurs_at')
                graph.add_edge(mut_id, f'aa:{name[0]}', relation='from_aa')
                graph.add_edge(mut_id, f'aa:{name[-1]}', relation='to_aa')
                # 高分单点的直接观测，不声称相比 WT 已证实改善。
                if len(mutations) == 1 and is_high:
                    graph.add_edge(mut_id, obs_id, relation='has_high_single_observation')

    def _observations(self, mutation):
        """沿 Mutation <-contains- Variant -has_fitness-> Observation 查询。"""
        mut_id = f'mutation:{mutation}'
        if mut_id not in self.graph:
            return []
        observations = []
        for variant in self.graph.predecessors(mut_id):
            if self.graph.nodes[variant]['type'] != 'Variant':
                continue
            obs_id = f"fitness:{self.graph.nodes[variant]['mutant']}"
            obs = self.graph.nodes[obs_id]
            observations.append({
                'variant': self.graph.nodes[variant]['mutant'],
                'mutation_count': self.graph.nodes[variant]['mutation_count'],
                'measured_fitness': round(obs['score'], 5),
                'high_top10': obs['high_top10'],
                'source': obs['source'],
            })
        return sorted(observations, key=lambda x: x['measured_fitness'], reverse=True)

    def _combo(self, mutant):
        """查询同时包含至少两个目标替换的已测高分变体。"""
        mutations = mutant.split(':')
        if len(mutations) < 2:
            return []
        counts = Counter()
        for name in mutations:
            node = f'mutation:{name}'
            if node in self.graph:
                counts.update(v for v in self.graph.predecessors(node)
                              if self.graph.nodes[v]['type'] == 'Variant')
        matches = []
        for variant, count in counts.items():
            obs = self.graph.nodes[f"fitness:{self.graph.nodes[variant]['mutant']}"]
            if count >= 2 and obs['high_top10']:
                matches.append({
                    'variant': self.graph.nodes[variant]['mutant'],
                    'shared_mutations': count,
                    'measured_fitness': round(obs['score'], 5),
                    'source': obs['source'],
                })
        return sorted(matches, key=lambda x: x['measured_fitness'], reverse=True)[:3]

    def evidence(self, mutant):
        # 组合候选可以复用单点的历史证据，不能把多点背景效应当成单点因果效应。
        mutations = []
        for name in mutant.split(':'):
            observations = self._observations(name)
            direct = [o for o in observations if o['mutation_count'] == 1]
            high = [o for o in observations if o['high_top10']]
            mutations.append({
                **self.mutation(name),
                'single_observation': direct[0] if direct else None,
                'high_variant_count': len(high),
                'high_variant_examples': high[:2],
            })
        return {
            'mutations': mutations,
            'related_high_combinations': self._combo(mutant),
            'evidence_limit': '高分关联并不证明突变相对野生型提高 Fitness',
        }

    def context(self, analysis):
        basic = super().context(analysis)
        # 小批量图谱查询摘要，而不是把全部节点/边放进 Prompt。
        top_singles = []
        for row in analysis['top_single_mutants'][:10]:
            name = row['mutant']
            obs = self._observations(name)
            single = next((o for o in obs if o['mutation_count'] == 1), None)
            top_singles.append({
                'mutation': name,
                'position': int(name[1:-1]),
                'single_measured_fitness': single['measured_fitness'] if single else None,
                'high_variant_count': sum(o['high_top10'] for o in obs),
                'high_combination_examples': [o for o in obs if o['high_top10'] and o['mutation_count'] > 1][:2],
            })
        high_variants = []
        for variant, attrs in self.graph.nodes(data=True):
            if attrs['type'] != 'Variant' or attrs['mutation_count'] < 2:
                continue
            obs = self.graph.nodes[f"fitness:{attrs['mutant']}"]
            if obs['high_top10']:
                high_variants.append({
                    'variant': attrs['mutant'],
                    'measured_fitness': round(obs['score'], 5),
                    'mutation_count': attrs['mutation_count'],
                })
        high_variants.sort(key=lambda o: o['measured_fitness'], reverse=True)
        basic['graph'] = {
            'selection': '仅 train：全部单点突变 + 训练集实测 Fitness 前10%的变体（取并集）',
            'statistics': self.stats,
            'top_single_evidence': top_singles,
            'top_high_multi_variants': high_variants[:8],
            'relations': [
                'AminoAcid -has_property-> Property（边属性 value）',
                'Mutation -occurs_at-> Position',
                'Variant -contains-> Mutation',
                'Variant -has_fitness-> FitnessObservation',
                'Mutation -has_high_single_observation-> FitnessObservation（仅直接高分单点观测）',
            ],
            'caveat': '没有可比的 WT 实测基线，不构造未经证实的 Mutation -improves-> Fitness 因果边',
        }
        return basic
