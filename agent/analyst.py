class DataAnalyst:
    def analyze(self, df, top_k=10, min_count=8, high_quantile=0.9):
        data = df.copy()
        data['mutation_count'] = data['mutant'].str.count(':') + 1
        data['is_high'] = data['DMS_score'] >= data['DMS_score'].quantile(high_quantile)
        fields = ['mutant', 'DMS_score', 'mutation_count']
        top = data.nlargest(top_k, 'DMS_score')[fields]
        singles = data[data['mutation_count'] == 1].nlargest(top_k, 'DMS_score')[fields]
        parts = data[['mutant', 'DMS_score', 'is_high']].assign(
            mutation=data['mutant'].str.split(':')
        ).explode('mutation')
        parts['position'] = parts['mutation'].str.extract(r'(\d+)').astype(int)

        def summarize(key):
            stats = parts.groupby(key).agg(
                n=('is_high', 'size'),
                mean_fitness=('DMS_score', 'mean'),
                high_rate=('is_high', 'mean')
            ).reset_index()
            stats['enrichment'] = stats['high_rate'] / data['is_high'].mean()
            return stats[stats['n'] >= min_count].sort_values(
                ['enrichment', 'n'], ascending=False
            ).head(top_k).round(3).to_dict('records')

        return {
            'top_variants': top.to_dict('records'),
            'top_single_mutants': singles.to_dict('records'),
            'promising_mutations': summarize('mutation'),
            'promising_sites': summarize('position'),
        }
