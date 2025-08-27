#!/bin/bash

echo "🚀 Creating Clean Repository Script"
echo "=================================="

# Step 1: Create a new directory for clean repository
echo "📁 Creating clean repository directory..."
mkdir -p ../multi_agent_ai_simulator_clean
cd ../multi_agent_ai_simulator_clean

# Step 2: Copy only the necessary files (excluding .git, venv, etc.)
echo "📋 Copying project files..."
cp -r ../multi_agent_ai_simulator/.streamlit . 2>/dev/null || true
cp -r ../multi_agent_ai_simulator/data . 2>/dev/null || true
cp -r ../multi_agent_ai_simulator/frontend . 2>/dev/null || true
cp -r ../multi_agent_ai_simulator/image . 2>/dev/null || true
cp -r ../multi_agent_ai_simulator/logs . 2>/dev/null || true
cp -r ../multi_agent_ai_simulator/notebooks . 2>/dev/null || true
cp -r ../multi_agent_ai_simulator/src . 2>/dev/null || true
cp -r ../multi_agent_ai_simulator/tests . 2>/dev/null || true

# Copy individual files
cp ../multi_agent_ai_simulator/.gitignore . 2>/dev/null || true
cp ../multi_agent_ai_simulator/DEPLOYMENT_GUIDE.md . 2>/dev/null || true
cp ../multi_agent_ai_simulator/README.md . 2>/dev/null || true
cp ../multi_agent_ai_simulator/multi_agent_ai_simulator.py . 2>/dev/null || true
cp ../multi_agent_ai_simulator/packages.txt . 2>/dev/null || true
cp ../multi_agent_ai_simulator/requirements.txt . 2>/dev/null || true
cp ../multi_agent_ai_simulator/test_deployment.py . 2>/dev/null || true

# Step 3: Initialize new git repository
echo "🔧 Initializing new Git repository..."
git init

# Step 4: Add all files
echo "📝 Adding files to repository..."
git add .

# Step 5: Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Multi-Agent AI Bidding Simulator - Clean Repository"

echo ""
echo "✅ Clean repository created successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Delete the old repository on GitHub"
echo "2. Create a new repository with a different name (e.g., 'Multi-Agent-AI-Bidding-Simulator-Clean')"
echo "3. Run these commands in the new directory:"
echo "   cd ../multi_agent_ai_simulator_clean"
echo "   git remote add origin https://github.com/RounitR/NEW_REPO_NAME.git"
echo "   git push -u origin main"
echo ""
echo "🎯 This will give you a completely clean repository with only your contributions!"
