# Contributing to Multi-Agent AI Bidding Simulator

Thank you for your interest in contributing to the Multi-Agent AI Bidding Simulator! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.9 or higher
- Git for version control
- OpenAI API key (for full functionality)

### Local Development Setup
```bash
# Clone the repository
git clone https://github.com/RounitR/Multi-Agent-AI-Bidding-System.git
cd multi_agent_ai_simulator

# Create virtual environment
python3 -m venv venv_py39
source venv_py39/bin/activate  # On Windows: venv_py39\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Run the application
streamlit run frontend/enhanced_app.py
```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Follow the existing code style and conventions
- Add appropriate comments and documentation
- Include tests for new functionality

### 3. Test Your Changes
```bash
# Run the enhanced system tests
python test_enhanced_system.py

# Test OpenAI integration
python test_openai.py

# Run the application locally
streamlit run frontend/enhanced_app.py
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "Add feature: brief description of changes"
```

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add type hints where appropriate
- Include docstrings for functions and classes

### Example Code Style
```python
def calculate_optimal_bid(
    market_threshold: float,
    cost_price: float,
    competitor_bids: List[float]
) -> Tuple[float, float]:
    """
    Calculate the optimal bid based on market conditions.
    
    Args:
        market_threshold: Maximum acceptable bid
        cost_price: Minimum viable bid
        competitor_bids: List of competitor bid amounts
        
    Returns:
        Tuple of (optimal_bid, confidence_score)
    """
    # Implementation here
    pass
```

### Frontend Code (Streamlit)
- Use consistent naming conventions
- Organize components logically
- Add appropriate error handling
- Ensure responsive design

## Testing Guidelines

### Unit Tests
- Test individual functions and methods
- Use descriptive test names
- Include edge cases and error conditions
- Maintain good test coverage

### Integration Tests
- Test component interactions
- Verify end-to-end functionality
- Test with different data scenarios

### Example Test Structure
```python
def test_enhanced_bidding_agent():
    """Test enhanced bidding agent functionality."""
    agent = EnhancedBiddingAgent("test_agent", "balanced")
    
    # Test bid generation
    bid = agent.generate_bid(1000, 500, [800, 900, 950])
    assert 500 <= bid <= 1000
    
    # Test learning
    agent.update_reward(state, action, True, 950)
    assert agent.total_reward > 0
```

## Documentation Guidelines

### Code Documentation
- Add docstrings to all functions and classes
- Include parameter descriptions and return types
- Provide usage examples where helpful

### README Updates
- Update README.md for new features
- Include screenshots for UI changes
- Update installation instructions if needed

### Technical Documentation
- Update TECHNICAL_DOCUMENTATION.md for architectural changes
- Document new algorithms and approaches
- Include performance considerations

## Pull Request Guidelines

### Before Submitting
1. **Test thoroughly**: Ensure all tests pass
2. **Update documentation**: Include relevant documentation changes
3. **Check code style**: Follow project conventions
4. **Review changes**: Self-review your changes

### Pull Request Description
- Provide a clear description of changes
- Include screenshots for UI changes
- Reference related issues
- List any breaking changes

### Example Pull Request
```markdown
## Description
Adds enhanced AI Assistant with context-aware responses and quick action buttons.

## Changes
- Implemented intelligent chat interface
- Added quick action buttons for common queries
- Enhanced keyword matching for better responses
- Added loading indicators and auto-clearing input

## Testing
- [x] All existing tests pass
- [x] New tests added for AI Assistant functionality
- [x] Manual testing completed

## Screenshots
[Include screenshots of new features]
```

## Issue Reporting

### Bug Reports
- Provide clear description of the issue
- Include steps to reproduce
- Add relevant error messages
- Specify environment details

### Feature Requests
- Describe the desired functionality
- Explain the use case
- Suggest implementation approach
- Consider impact on existing features

## Code Review Process

### Review Checklist
- [ ] Code follows project style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes introduced
- [ ] Performance impact considered
- [ ] Security implications reviewed

### Review Comments
- Be constructive and helpful
- Suggest specific improvements
- Ask clarifying questions
- Provide context for suggestions

## Release Process

### Version Management
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- Update CHANGELOG.md for each release
- Tag releases in Git
- Update version numbers in code

### Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version numbers updated
- [ ] Release notes prepared
- [ ] Deployment tested

## Contact and Support

### Questions and Discussions
- Use GitHub Issues for questions
- Create discussions for feature ideas
- Join project discussions for collaboration

### Getting Help
- Check existing documentation
- Review closed issues for similar problems
- Ask specific, detailed questions
- Provide context and error details

## Recognition

### Contributors
- All contributors will be recognized in the project
- Significant contributions will be highlighted
- Contributors will be listed in README.md

### Contribution Types
- Code contributions
- Documentation improvements
- Bug reports and fixes
- Feature suggestions
- Testing and feedback

Thank you for contributing to the Multi-Agent AI Bidding Simulator! 🚀
