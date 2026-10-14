import re

PATTERN = re.compile(r'([ACDEFGHIKLMNPQRSTVWY])([1-9]\d*)([ACDEFGHIKLMNPQRSTVWY])')


def infer_wild_type(df):
    row = df.iloc[0]
    sequence = list(row['mutated_sequence'])
    for mutation in row['mutant'].split(':'):
        old, position, new = PATTERN.fullmatch(mutation).groups()
        index = int(position) - 1
        if sequence[index] != new:
            raise ValueError('突变注释与参考序列不匹配')
        sequence[index] = old
    return ''.join(sequence)


class MutationDesigner:
    def design(self, hypotheses, wild_type, known, max_mutations=5):
        seen, candidates, rejected = set(known), [], []
        for item in hypotheses:
            mutations = item['mutations']
            if not 1 <= len(mutations) <= max_mutations:
                rejected.append({'mutations': mutations, 'reason': 'mutation_count'})
                continue
            sequence = list(wild_type)
            positions = set()
            valid = True
            for mutation in mutations:
                match = PATTERN.fullmatch(mutation)
                if not match:
                    valid = False
                    break
                old, pos, new = match.groups()
                index = int(pos) - 1
                if index >= len(wild_type) or wild_type[index] != old or old == new or index in positions:
                    valid = False
                    break
                sequence[index] = new
                positions.add(index)
            if not valid:
                rejected.append({'mutations': mutations, 'reason': 'invalid_mutation'})
                continue
            mutated_sequence = ''.join(sequence)
            if mutated_sequence in seen:
                rejected.append({'mutations': mutations, 'reason': 'duplicate'})
                continue
            seen.add(mutated_sequence)
            ordered = sorted(mutations, key=lambda x: int(PATTERN.fullmatch(x).group(2)))
            candidates.append({
                'mutant': ':'.join(ordered),
                'mutated_sequence': mutated_sequence,
                'mutation_count': len(ordered),
                'strategy': item['strategy'],
                'reason': item['reason'],
                'risk': item['risk'],
            })
        self.rejections = rejected
        return candidates
