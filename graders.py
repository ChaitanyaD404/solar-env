def grade(total_reward, max_possible_reward):
    if max_possible_reward == 0:
        return 0.0

    score = total_reward / max_possible_reward

    # clamp between 0 and 1
    if score > 1:
        score = 1.0
    if score < 0:
        score = 0.0

    return score
