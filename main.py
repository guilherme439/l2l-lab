from SCS_Game import SCS_Game


def main():
    config_path = "/home/guilherme/Documents/Code/Personal/RL-SCS/RL-SCS/Game_configs/randomized_config_5.yml"
    
    game = SCS_Game(config_path)
    
    print(f"Game initialized successfully!")
    print(f"Board size: {game.rows}x{game.columns}")
    print(f"Current player: {game.current_player}")
    print(f"Current turn: {game.current_turn}")
    
    return game


if __name__ == "__main__":
    game = main()

