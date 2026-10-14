from Bio.Align import substitution_matrices
import pandas as pd

from agent.designer import PATTERN


class KnowledgeBase:
    """只提供生物学参考信息，不参与硬筛选或修改 MLP 分数。"""

    def __init__(self, csv_path):
        table = pd.read_csv(csv_path).set_index('AA')
        self.properties = table.to_dict('index')
        self.columns = list(table.columns)
        self.blosum = substitution_matrices.load('BLOSUM62')

    def mutation(self, name):
        old, position, new = PATTERN.fullmatch(name).groups()
        changes = {
            column: round(self.properties[new][column] - self.properties[old][column], 3)
            for column in self.columns
        }
        return {
            'mutation': name,
            'position': int(position),
            'BLOSUM62': int(self.blosum[old, new]),
            'property_delta_to_minus_from': changes,
        }

    def evidence(self, mutant):
        return [self.mutation(name) for name in mutant.split(':')]

    def context(self, analysis):
        amino_acids = [
            {'AA': name, **properties}
            for name, properties in self.properties.items()
        ]
        conservative = {
            name: [
                {'to': other, 'score': int(self.blosum[name, other])}
                for other in self.properties
                if other != name and self.blosum[name, other] > 0
            ]
            for name in self.properties
        }
        historical = [
            {
                'mutant': row['mutant'],
                'measured_fitness': row['DMS_score'],
                'knowledge': self.evidence(row['mutant']),
            }
            for row in analysis['top_single_mutants']
        ]
        return {
            'property_source': 'data/GFP_properties.xlsx (data/aa_props.csv 为等值副本)',
            'properties': amino_acids,
            'property_columns': self.columns,
            'conservative_options_BLOSUM62_positive': conservative,
            'high_fitness_single_mutants_from_training': historical,
        }
