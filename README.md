# AI-Powered RPG Level Generator with Auto Fine-tuning

An intelligent RPG level generator that uses GPT-4.1 initially and automatically fine-tunes TinyLlama based on performance metrics. Built with JAC (Jac Programming Language) and Pygame.

## Features

- **AI-Powered Level Generation**: Uses GPT-4.1 to generate RPG levels and maps
- **Automatic Fine-tuning**: Collects training data and fine-tunes TinyLlama when accuracy threshold is met
- **Model Swapping**: Automatically switches to fine-tuned model for improved performance
- **Quality Evaluation**: Evaluates generated content and collects high-quality examples for training
- **Interactive Gameplay**: Pygame-based RPG with player movement, enemies, and obstacles
- **Visual Map Display**: Real-time ASCII and graphical representation

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [JAC Programming Language](https://www.jac-lang.org/)
- OpenAI API key (for GPT-4.1)

## Installation

### 1. Install uv Package Manager

```bash
# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository

```bash
git clone <repository-url>
cd jac_impl
```

### 3. Install Dependencies with uv

```bash
# Create virtual environment and install all dependencies from pyproject.toml
uv sync

# Or manually create venv and install
uv venv
uv pip install -e .
```

#### Dependencies (from pyproject.toml)

The project automatically installs:

```toml
# Core JAC and AI/ML libraries
jaclang>=0.8.5
mtllm>=0.4.0
transformers>=4.55.4
torch>=2.8.0
datasets>=4.0.0
peft>=0.17.1
accelerate>=1.10.0
trl>=0.21.0

# Game libraries
pygame>=2.6.1

# Data processing
numpy>=2.3.2
pandas>=2.3.2

# Optional: GPU acceleration
bitsandbytes>=0.47.0
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LiteLLM Configuration
LITELLM_LOG=INFO
TRANSFORMERS_VERBOSITY=info
TOKENIZERS_PARALLELISM=false
```

## Usage

### 1. Activate Virtual Environment

```bash
# uv automatically manages the virtual environment
# But you can manually activate if needed:

# On Linux/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### 2. Start the Complete RPG Game

```bash
# Navigate to the game directory
cd jac_impl_6

# Run the main RPG game
jac run main.jac
```

### 3. Run Level Generator Only

```bash
# Navigate to utils directory
cd jac_impl_6/utils

# Run just the AI level generator
jac run level_generator.jac
```

## Game Structure

### Project Layout
```
jac_impl/
├── jac_impl_6/                    # Main game directory
│   ├── main.jac                   # Game entry point
│   ├── game_obj.jac               # Game objects and mechanics
│   ├── sprites.jac                # Player, enemy, and item sprites
│   ├── settings/
│   │   ├── config.jac             # Game configuration
│   │   └── map.jac                # Map definitions
│   ├── utils/
│   │   ├── level_generator.jac    # AI level generator with auto fine-tuning
│   │   └── rpg_level_training_data.json  # Generated training data
│   ├── game_obj.impl/             # Game object implementations
│   ├── main.impl/                 # Main game implementations
│   ├── sprites.impl/              # Sprite implementations
│   └── tinyllama-rpg-finetuned/   # Fine-tuned model output
├── fonts/                         # Game fonts
├── img/                           # Game assets and images
├── pyproject.toml                 # Project dependencies
├── .env                           # Environment variables
└── README.md                      # This file
```

## How It Works

### 1. Game Flow
- **Start Screen**: Initial game interface
- **Level Generation**: AI creates new levels dynamically
- **Gameplay**: Player navigates through generated levels
- **Progression**: Completing levels triggers new AI generation

### 2. AI Level Generation Process

#### Initial Phase (GPT-4.1)
- Generates RPG levels using GPT-4.1
- Creates maps with walls, enemies, and obstacles
- Evaluates generation quality (0-1 score)
- Collects high-quality examples (≥0.7 score) for training

#### Training Data Collection
- Monitors generation accuracy
- Requires minimum 20 training examples
- Accuracy threshold: 0.85 (85% success rate)

#### Automatic Fine-tuning
- Triggers when accuracy threshold is met
- Fine-tunes TinyLlama using LoRA (Low-Rank Adaptation)
- Saves model to `./tinyllama-rpg-finetuned/`

#### Model Swapping
- Automatically switches to fine-tuned TinyLlama
- Continues level generation with improved performance
- Maintains performance metrics and history

## Game Controls

### Gameplay Elements
- **P**: Player starting position
- **E**: Enemy positions  
- **B**: Walls and obstacles
- **.**: Empty walkable tiles

### Controls (Standard Pygame)
- **Arrow Keys**: Move player
- **WASD**: Alternative movement
- **ESC**: Pause/Menu
- **Space**: Action/Attack

## Configuration

### Level Parameters
- **Difficulty**: 1-10 (increases every 2 levels)
- **Size**: 20x20 tiles (configurable)
- **Enemies**: Based on difficulty level
- **Walls**: Random placement with accessibility checks
- **Time Limit**: 300 seconds default

## Troubleshooting

### Common Issues

1. **JAC Command Not Found**
   ```bash
   uv pip install jaclang
   # Ensure .venv is activated
   ```

2. **OpenAI API Key Error**
   - Ensure `.env` file contains valid `OPENAI_API_KEY`
   - Check API key permissions and billing

3. **CUDA/GPU Issues**
   ```bash
   # Install CUDA-compatible PyTorch
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Memory Issues During Fine-tuning**
   - Reduce batch size in training arguments
   - Use CPU training if GPU memory is insufficient
   - Close other applications to free RAM

5. **Pygame Display Issues**
   ```bash
   # On Linux, may need additional packages
   sudo apt-get install python3-pygame
   
   # On Windows, ensure DirectX is updated
   # On macOS, ensure Xcode command line tools are installed
   ```

6. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   uv sync
   
   # Check virtual environment
   which python  # Should point to .venv/bin/python
   ```

### Performance Tips

- **GPU Acceleration**: Install CUDA-compatible PyTorch for faster fine-tuning
- **Batch Size**: Adjust based on available memory  
- **Training Data**: More high-quality examples improve fine-tuned model performance
- **Game Performance**: Close unnecessary applications for smoother gameplay

## Development

### Adding Features

1. **Custom Level Types**: Modify `Level` object in `level_generator.jac`
2. **New AI Models**: Update model initialization in `init_llm()`
3. **Enhanced Evaluation**: Improve `evaluate_generation_quality()` method
4. **New Game Mechanics**: Add to `game_obj.jac` and corresponding implementations

### Running Tests

```bash
# Run level generator tests
cd jac_impl_6/utils
jac run level_generator.jac

# Test complete game flow
cd jac_impl_6
jac run main.jac
```

### Code Structure

- **JAC Files**: Main game logic and AI integration
- **Implementation Files**: Detailed game mechanics
- **Settings**: Configuration and map definitions
- **Utils**: AI level generation and utilities

## API Keys and Configuration

### Required API Keys
- **OpenAI API Key**: For GPT-4.1 access (required)

### Optional Configuration
- **LiteLLM Settings**: Advanced model routing
- **CUDA Settings**: GPU acceleration
- **Transformers Cache**: Model caching location

## License

[Add your license information here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both level generator and full game
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review JAC documentation: https://www.jac-lang.org/
3. Review Pygame documentation: https://pygame.org/docs/
4. Create an issue in the repository

---

**Note**: This project requires an active OpenAI API key with GPT-4.1 access. Fine-tuning will automatically begin once the accuracy threshold is met with sufficient training data. The game combines traditional RPG mechanics with cutting-edge AI for dynamic level generation.
