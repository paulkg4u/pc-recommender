import pandas as pd
import random

# Load components data
components_df = pd.read_csv('components.csv')

# Define a function to generate a new build
def generate_build(start_id, i):
    cpu_options = components_df[components_df['component_type'] == 'CPU']['component_id']
    gpu_options = components_df[components_df['component_type'] == 'GPU']['component_id']
    ram_options = components_df[components_df['component_type'] == 'RAM']['component_id']

    if cpu_options.empty or gpu_options.empty or ram_options.empty:
        print("Not enough components to generate a build.")
        return None

    cpu = random.choice(cpu_options.to_list())
    gpu = random.choice(gpu_options.to_list())
    ram = random.choice(ram_options.to_list())
    
    # Calculate total price and performance score
    cpu_row = components_df[components_df['component_id'] == cpu].iloc[0]
    gpu_row = components_df[components_df['component_id'] == gpu].iloc[0]
    ram_row = components_df[components_df['component_id'] == ram].iloc[0]
    
    total_price = cpu_row['price'] + gpu_row['price'] + ram_row['price']
    total_performance_score = int((cpu_row['performance_score'] + gpu_row['performance_score'] + ram_row['performance_score']) / 3)
    
    # Determine optimal use case
    if total_performance_score >= 8:
        optimal_for = 'gaming'
    elif total_performance_score >= 6:
        optimal_for = 'video_editing'
    elif total_performance_score >= 4:
        optimal_for = 'programming'
    else:
        optimal_for = 'daily_use'
    
    return {
        'build_id': start_id + i,
        'cpu_id': cpu,
        'gpu_id': gpu,
        'ram_id': ram,
        'price': total_price,
        'total_performance_score': total_performance_score,
        'optimal_for': optimal_for
    }
# Generate new builds
start_id = len(pd.read_csv('builds.csv')) + 1
new_builds = [generate_build(start_id, i) for i in range(10)]

# Convert to DataFrame and save to CSV
new_builds_df = pd.DataFrame(new_builds)
with open('builds.csv', 'a', newline='\n') as f:
    new_builds_df.to_csv(f, header=False, index=False)