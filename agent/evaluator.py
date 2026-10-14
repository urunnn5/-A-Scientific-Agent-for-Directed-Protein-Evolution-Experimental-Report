class FitnessEvaluator:
    def __init__(self, predictor):
        self.predictor = predictor

    def evaluate(self, candidates, top_k):
        if not candidates:
            return {'all_scored': [], 'top_k': []}
        scores = self.predictor.predict([c['mutated_sequence'] for c in candidates])
        scored = [
            {**candidate, 'predicted_fitness': float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x['predicted_fitness'], reverse=True)
        for rank, candidate in enumerate(scored, 1):
            candidate['rank'] = rank
        return {'all_scored': scored, 'top_k': scored[:top_k]}
