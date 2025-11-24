from rl_scs import SCS_Game


def main():
    config_path = "/home/guilherme/Documents/Code/Personal/RL-SCS/RL-SCS/src/example_configurations/randomized_config_5.yml"
    
    game = SCS_Game(config_path)
    
    print(f"Game initialized successfully!")
    print(f"Board size: {game.rows}x{game.columns}")
    print(f"Current agent: {game.agent_selection}")
    print(f"Current turn: {game.current_turn}")
    
    return game


if __name__ == "__main__":
    game = main()

