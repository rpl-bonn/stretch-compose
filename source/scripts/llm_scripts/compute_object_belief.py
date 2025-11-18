import json
import math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# === Tunable Parameters ===
# Observation confidence
OBS_HIT = 0.8
OBS_MISS = 0.2
# Confidence update multipliers
CONFIDENCE_DECAY = 0.95
CONFIDENCE_BOOST = 1.2
CONFIDENCE_PENALTY = 0.7
CONFIDENCE_TOP_N = 3
# Probability and smoothing
PROB_PRUNE_THRESHOLD = 1e-6
LAPLACE_ALPHA = 1.0
# New location handling
NEW_LOCATION_CONFIDENCE = 0.3  # Initial confidence for newly discovered locations

# === Math Helpers (Log-odds for numerical stability) ===
def logodds(p):
    """Converts a probability to log-odds. Adds epsilon to prevent log(0)."""
    p = max(min(p, 1.0 - 1e-9), 1e-9)  # Clamp to avoid log(0) or log(1)
    return math.log(p / (1.0 - p))

def sigmoid(lo):
    """Converts log-odds to a probability."""
    return 1.0 / (1.0 + math.exp(-lo))

# === Belief Representation ===
class ObjectBelief:
    """Represents the belief state for a single object."""
    def __init__(self, obj_id, possible_locations):
        self.object_id = obj_id
        # Internal state is stored in log-odds for numerical stability
        self.log_odds = {loc: logodds(1.0 / len(possible_locations)) for loc in possible_locations}
        self.confidence = 0.5  # Start with neutral confidence
        self.known_locations = set(possible_locations)

    def add_new_location(self, new_loc):
        """Dynamically add a new location to the belief state."""
        if new_loc not in self.known_locations:
            # Initialize with low probability (high uncertainty)
            self.log_odds[new_loc] = logodds(NEW_LOCATION_CONFIDENCE / (len(self.known_locations) + 1))
            self.known_locations.add(new_loc)
            # Renormalize existing beliefs slightly to account for new location
            renorm_factor = (1.0 - NEW_LOCATION_CONFIDENCE)
            for loc in self.log_odds:
                if loc != new_loc:
                    current_p = sigmoid(self.log_odds[loc])
                    self.log_odds[loc] = logodds(current_p * renorm_factor)

    def get_prob_dist(self):
        """Converts the internal log-odds to a normalized probability distribution."""
        probs = {loc: sigmoid(lo) for loc, lo in self.log_odds.items()}
        Z = sum(probs.values())
        if Z == 0:
            return {loc: 1.0 / len(probs) for loc in probs}
        return {loc: p / Z for loc, p in probs.items()}

# === Transition Model Learning (Correctly implemented to be used incrementally) ===
def learn_transition_model(transition_counts, all_locations):
    """
    Converts a transition count matrix to a probability matrix using Laplace smoothing.
    This function is now used to re-calculate the model after an incremental update to counts.
    """
    transition_probs = defaultdict(dict)
    for current_loc in all_locations:
        total_transitions = sum(transition_counts[current_loc].values()) + LAPLACE_ALPHA * len(all_locations)
        for next_loc in all_locations:
            count = transition_counts[current_loc].get(next_loc, 0)
            transition_probs[current_loc][next_loc] = (count + LAPLACE_ALPHA) / total_transitions
    
    return dict(transition_probs)

# === Prediction Step (Corrected to sum probabilities, not log-odds) ===
def predict_next_belief(current_belief, transition_model, all_locations):
    """
    Predicts the next belief state using the transition model and current belief.
    """
    prob_dist = current_belief.get_prob_dist()
    predicted_probs = defaultdict(float)

    for current_loc, current_prob in prob_dist.items():
        if current_prob > 0:
            transition_probs = transition_model.get(current_loc, {})
            for next_loc, transition_prob in transition_probs.items():
                predicted_probs[next_loc] += current_prob * transition_prob
    
    # Create new belief object
    new_belief = ObjectBelief(current_belief.object_id, all_locations)
    # Convert predicted probabilities back to log-odds
    new_belief.log_odds = {loc: logodds(predicted_probs[loc]) for loc in predicted_probs}
    new_belief.confidence = current_belief.confidence
    new_belief.known_locations = current_belief.known_locations.copy()
    return new_belief

# === Observation Update (Corrected likelihood calculation) ===
def update_belief_with_observation(pred_belief, observed_loc):
    """
    Performs a Bayesian update on the belief state using a new observation.
    """
    updated_belief = ObjectBelief(pred_belief.object_id, list(pred_belief.log_odds.keys()))
    updated_belief.known_locations = pred_belief.known_locations.copy()
    
    # Handle new locations dynamically
    if observed_loc is not None and observed_loc not in pred_belief.known_locations:
        updated_belief.add_new_location(observed_loc)
    
    predicted_prob_dist = pred_belief.get_prob_dist()
    
    if observed_loc is None:
        updated_belief.log_odds = pred_belief.log_odds
        updated_belief.confidence = pred_belief.confidence * CONFIDENCE_DECAY
        return updated_belief
    
    # Bayesian update in log-odds space (add likelihood to prior)
    for location in updated_belief.known_locations:
        lo_prior = pred_belief.log_odds.get(location, logodds(PROB_PRUNE_THRESHOLD))
        if location == observed_loc:
            lo_likelihood = logodds(OBS_HIT)
        else:
            lo_likelihood = logodds(OBS_MISS)
            
        updated_belief.log_odds[location] = lo_prior + lo_likelihood

    # Prune locations below a probability threshold
    updated_belief.log_odds = {
        loc: lo for loc, lo in updated_belief.log_odds.items()
        if sigmoid(lo) > PROB_PRUNE_THRESHOLD
    }
    
    # Update confidence based on prediction accuracy
    if predicted_prob_dist:
        predicted_top_loc = max(predicted_prob_dist, key=predicted_prob_dist.get)
        if observed_loc == predicted_top_loc:
            updated_belief.confidence = min(1.0, pred_belief.confidence * CONFIDENCE_BOOST)
        elif observed_loc not in list(predicted_prob_dist.keys())[:CONFIDENCE_TOP_N]:
            updated_belief.confidence = max(0.1, pred_belief.confidence * CONFIDENCE_PENALTY)
        else:
            updated_belief.confidence = pred_belief.confidence
    
    return updated_belief

# === Entropy for Uncertainty ===
def entropy(prob_dist):
    """Calculates the Shannon entropy of a probability distribution."""
    return -sum(p * math.log(p + 1e-12) for p in prob_dist.values() if p > 0)

# === Driver (Corrected to be truly incremental and efficient) ===
def run_incremental_learning(obj_file, graph_file):
    """
    Main driver function to run the incremental belief tracking simulation.
    """
    try:
        obj_data = json.loads(Path(obj_file).read_text())
        graph_data = json.loads(Path(graph_file).read_text())
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Please check the file path: {e}")
        return

    # Map immovable ids to labels
    id_to_label = {nid: lbl for nid, lbl in zip(graph_data["node_ids"], graph_data["node_labels"])}

    # Build location keys from object history
    historical_transitions = []
    for h in obj_data["history"]:
        loc = (h["furniture_id"], h["drawer_id"])
        historical_transitions.append((h["date"], loc))

    # Initialize with locations from the first observation
    initial_locations = list({historical_transitions[0][1]})
    belief = ObjectBelief(obj_data["id"], initial_locations)
    transition_counts = defaultdict(lambda: defaultdict(int))
    
    entropy_over_days = []
    prior_entropies = []
    posterior_entropies = []
    # Data for the new plot
    predicted_locations_over_days = []
    observed_locations_over_days = []
    
    print("Starting incremental belief tracking simulation...")
    
    # Handle the first day separately for initialization
    first_day_loc = historical_transitions[0][1]
    belief = update_belief_with_observation(belief, first_day_loc)
    first_probs = belief.get_prob_dist()
    first_ent = entropy(first_probs)
    entropy_over_days.append(first_ent)
    prior_entropies.append(first_ent)
    posterior_entropies.append(first_ent)
    
    predicted_locations_over_days.append(first_day_loc)
    observed_locations_over_days.append(first_day_loc)
    
    print(f"\n=== Day 1 ({historical_transitions[0][0]}) ===")
    print("Initial Observed:", f"{first_day_loc[0]} ({id_to_label.get(first_day_loc[0],'?')})")
    print("Initial Belief:", {f"{lid} ({id_to_label.get(lid, '?')})": round(p,3) for lid,p in first_probs.items()})
    print("Initial Entropy:", round(first_ent,3))
    print(f"Confidence: {round(belief.confidence,3)}")
    
    # Process remaining days incrementally
    for day_index in range(1, len(historical_transitions)):
        day, true_loc = historical_transitions[day_index]
        prev_loc = historical_transitions[day_index - 1][1]
        
        # Incrementally update transition counts
        transition_counts[prev_loc][true_loc] += 1
        all_known_locations = belief.known_locations | {true_loc}
        transition_model = learn_transition_model(transition_counts, all_known_locations)
        
        # Predict next belief (prior)
        prior_belief = predict_next_belief(belief, transition_model, all_known_locations)
        prior_probs = prior_belief.get_prob_dist()
        prior_ent = entropy(prior_probs)
        prior_entropies.append(prior_ent)
        
        # Update belief with observation (posterior)
        belief = update_belief_with_observation(prior_belief, true_loc)
        posterior_probs = belief.get_prob_dist()
        posterior_ent = entropy(posterior_probs)
        posterior_entropies.append(posterior_ent)
        entropy_over_days.append(posterior_ent)
        
        # Collect data for the new plot
        predicted_top_loc = max(prior_probs, key=prior_probs.get) if prior_probs else None
        predicted_locations_over_days.append(predicted_top_loc)
        observed_locations_over_days.append(true_loc)
        
        # Calculate information gain
        info_gain = prior_ent - posterior_ent
        
        # Print readable output
        readable_prior = {f"{lid} ({id_to_label.get(lid, '?')})": round(p,3) for lid,p in prior_probs.items()}
        readable_posterior = {f"{lid} ({id_to_label.get(lid, '?')})": round(p,3) for lid,p in posterior_probs.items()}
        
        print(f"\n=== Day {day_index + 1} ({day}) ===")
        print("True Location:", f"{true_loc[0]} ({id_to_label.get(true_loc[0],'?')})")
        print("Prior Belief:", readable_prior)
        print("Posterior Belief:", readable_posterior)
        print(f"Uncertainty: Prior={round(prior_ent,3)}, Posterior={round(posterior_ent,3)}")
        print(f"Information Gain: {round(info_gain,3)}")
        print(f"Confidence: {round(belief.confidence,3)}")

    # Final prediction for next day
    all_known_locations = belief.known_locations
    transition_model = learn_transition_model(transition_counts, all_known_locations)
    final_prior = predict_next_belief(belief, transition_model, all_known_locations)
    final_probs = final_prior.get_prob_dist()
    readable_final = {f"{lid} ({id_to_label.get(lid, '?')})": round(p,3) for lid,p in final_probs.items()}
    
    print("\n=== Final Day Prediction (Next Step) ===")
    print("Next-location distribution:", readable_final)
    print("Uncertainty (entropy):", round(entropy(final_probs), 3))

    # === Plotting ===
    plt.figure(figsize=(10, 6))
    
    # Plot 1: Entropy
    ax1 = plt.subplot(2, 1, 1)
    days = range(1, len(entropy_over_days) + 1)
    ax1.plot(days, prior_entropies, 'r--', label='Prior Entropy (Prediction)')
    ax1.plot(days, posterior_entropies, 'b-', label='Posterior Entropy (After Update)')
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Entropy (uncertainty)")
    ax1.set_title("Uncertainty of Belief over Days")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Predicted vs Observed Location
    ax2 = plt.subplot(2, 1, 2)
    observed_y = [loc[0] for loc in observed_locations_over_days]
    predicted_y = [loc[0] if loc else None for loc in predicted_locations_over_days]
    
    # Get all unique locations for y-axis ticks
    all_locs_ids = list(set(observed_y) | set(predicted_y))
    all_locs_ids.sort()
    
    ax2.plot(days, observed_y, 'o-', color='blue', label='Observed Location')
    ax2.plot(days, predicted_y, 'x--', color='red', label='Predicted Top Location')
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Furniture ID")
    ax2.set_title("Predicted vs Observed Location")
    ax2.set_yticks(all_locs_ids)
    ax2.set_yticklabels([f"{loc_id} ({id_to_label.get(loc_id, '?')})" for loc_id in all_locs_ids])
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Example run
run_incremental_learning(
    "/home/rohit/workspace/ros2/stretch-compose/data/output_household/messy_single/seen/2025_02_26/st_graphs/object_102.json",
    "/home/rohit/workspace/ros2/stretch-compose/data/output_household/messy_single/seen/2025_02_26/graph.json"
)