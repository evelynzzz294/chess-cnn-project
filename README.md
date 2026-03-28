Chess Positional Evaluation with CNNs and Feature Engineering
Learning and evaluating positional structure in chess using convolutional neural networks combined with handcrafted positional features.

Overview
This project investigates how structured spatial patterns in chess positions can be modeled and evaluated using a hybrid approach: a CNN trained on FEN-encoded board states, augmented by interpretable handcrafted positional features. The system integrates with Stockfish for benchmarking and provides a Kivy-based interactive GUI for real-time move analysis.
The core insight is that while neural networks capture implicit spatial dependencies in board structure, handcrafted heuristics provide transparent, interpretable signals — and combining both improves evaluation consistency over either alone.

Key Insight
This project explores a question central to quantitative modeling: not all observable patterns are predictive.
Neural networks can learn spatial structure in chess positions, but learned representations alone don't always translate into consistent evaluation signals. Raw engine scores (centipawn) capture tactical sharpness but miss structural qualities like piece coordination or pawn weaknesses that matter over a longer horizon.
By combining a CNN's learned board representations with interpretable domain features, this project demonstrates that hybrid models can better distinguish meaningful structure from noise — producing evaluations that are more robust than either component alone.

Features
CNN-based position classification trained on FEN board encodings
Balanced batch training to handle class imbalance across position types
Hybrid positional scoring combining CNN output with engineered features
Stockfish integration for top-move retrieval and centipawn benchmarking
Interactive Kivy GUI with move arrows, history navigation, and real-time AI suggestions
Misclassification logging to capture and inspect model errors by epoch

Architecture
CNN Model (learn_2_FENs_Single_V010125debugv1_Colab.py)
Board positions are encoded as (8, 8, 12) tensors — one channel per piece type per color (6 white + 6 black, one-hot encoded). The network uses a dual-scale convolutional design:
Input (8×8×12)
→ Conv2D(128, 3×3) + BN + ReLU + MaxPool
→ Conv2D(128, 7×7) + BN + ReLU + MaxPool   ← wide receptive field for long-range patterns
→ Conv2D(256, 3×3) + BN + ReLU
→ Conv2D(256, 7×7) + BN + ReLU
→ Flatten → Dense(256) + Dropout(0.3)
→ Dense(128) → Output (softmax)
The 7×7 kernels are a deliberate design choice to capture board-spanning spatial relationships (e.g., open files, diagonals, long-range piece activity).
Balanced Batch Generator
A custom BalancedBatchGenerator (inheriting from keras.utils.Sequence) addresses class imbalance by oversampling minority position classes within each batch. Each batch contains an equal number of samples from every class, with minority classes upsampled via random replacement.
Positional Scoring (kivy_chess_HAI_v3.py)
A handcrafted position_score function evaluates each candidate move's resulting position across six signal categories: queen/rook/knight/bishop mobility and coverage, pawn structure penalties (isolated islands, doubled pawns), and center control (attackers on d4/d5/e4/e5). Each is independently weighted and summed. See position_score() in kivy_chess_HAI_v3.py for full weights and implementation.
composite_score = CP_score + pscore × pfactor   (pfactor = 15)
The HLAI (Hybrid Learning AI) picks the move with the highest composite score from Stockfish's top 3 candidates.

Project Structure
├── kivy_chess_HAI_v3.py               # Kivy GUI + positional scoring + HLAI engine
├── learn_2_FENs_Single_V010125debugv1_Colab.py  # CNN training pipeline (Colab)
├── positional_FEN_CS1.txt             # FEN dataset with position labels (not included)
├── stockfish/                         # Stockfish binary (not included, see Setup)
└── saved_model/
    └── group_classifier_model.keras   # Trained model checkpoint (generated after training)

Setup
Requirements
python >= 3.9
tensorflow >= 2.10
numpy
scikit-learn
matplotlib
python-chess
kivy
stockfish (binary)
Install Python dependencies:
bash
pip install tensorflow numpy scikit-learn matplotlib python-chess kivy
Stockfish
Download the appropriate Stockfish binary for your platform from stockfishchess.org and update the engine path in kivy_chess_HAI_v3.py:
python
self.engine = chess.engine.SimpleEngine.popen_uci(r"path/to/stockfish")

Training
Training is designed to run in Google Colab with GPU acceleration.
Upload learn_2_FENs_Single_V010125debugv1_Colab.py and your FEN dataset to Colab.
Set the dataset path:
python
  file_path = '/content/positional_FEN_CS1.txt'
Run the script. The model trains for up to 20 epochs with early stopping (patience=5, monitoring val_loss).
Starting at epoch 5, misclassified FENs are logged to /content/label_{label}_epoch_{N}.txt for error analysis.
The trained model is saved to /content/group_classifier_model.keras.
FEN Data Format
Each line in the dataset file should follow:
<FEN string> <label>
Where label is an integer class representing the position type or quality group.

Running the GUI
bash
python kivy_chess_HAI_v3.py
Interface
Click a piece to select it; legal moves are highlighted
AI Pick — Stockfish's top move (pure engine evaluation)
HLAI Pick — Hybrid move selected by composite score (engine + positional heuristics)
Back / Next — Navigate move history
Move arrows are drawn on the board for both suggestions

How the Hybrid Evaluation Works
For each position, Stockfish returns its top 3 candidate moves with centipawn scores. Each candidate is then scored by the handcrafted position_score function on the resulting board state. The composite score is:
tscore = CP + pscore × 15
The move with the highest tscore becomes the HLAI recommendation. When Stockfish's top move is dominant (margin > 20 CP), it is selected directly; otherwise the composite score arbitrates.

Limitations & Known Issues
The Stockfish binary path is currently hardcoded for a Windows environment — update before running on other platforms.
Training data (positional_FEN_CS1.txt) is not included in this repository.
The CNN is trained on position classification (grouping), not direct move quality regression.
pfactor = 15 is hand-tuned; optimal weighting may vary by game phase.

Tech Stack
Component
Technology
Neural network
TensorFlow / Keras
Data processing
NumPy, scikit-learn
Chess logic
python-chess
Engine interface
Stockfish (UCI)
GUI
Kivy
Training environment
Google Colab


Acknowledgments
Positional evaluation heuristics draw on classical chess theory (center control, piece activity, pawn structure). CNN architecture choices were informed by the spatial structure of the chess board and the need to capture both local tactics and long-range strategic patterns.
While developed in the context of chess, this framework reflects a broader modeling principle: combining learned representations with domain-specific signals to improve robustness in structured, noisy environments.


