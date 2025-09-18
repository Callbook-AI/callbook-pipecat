#!/usr/bin/env python3
"""
Simple test script to verify the Gladia optimized aggregators work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from gladia_optimized_aggregator import create_gladia_optimized_aggregators

def test_aggregator_creation():
    """Test that the optimized aggregators can be created successfully."""
    print("Testing aggregator creation...")
    
    # Create a basic OpenAI LLM context
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    context = OpenAILLMContext(messages)
    
    # Test creating the optimized aggregators
    try:
        aggregator_pair = create_gladia_optimized_aggregators(
            context,
            user_kwargs={
                'aggregation_timeout': 0.05,
                'bot_interruption_timeout': 0.1,
                'immediate_mode': True
            },
            assistant_kwargs={
                'expect_stripped_words': True,
                'min_chars_for_streaming': 15,
                'streaming_enabled': True
            }
        )
        
        print("✅ Aggregator pair created successfully")
        
        # Test that the aggregator pair has the expected methods
        user_aggregator = aggregator_pair.user()
        assistant_aggregator = aggregator_pair.assistant()
        
        print("✅ User aggregator retrieved successfully")
        print("✅ Assistant aggregator retrieved successfully")
        
        # Test basic properties
        print(f"User aggregator type: {type(user_aggregator).__name__}")
        print(f"Assistant aggregator type: {type(assistant_aggregator).__name__}")
        
        # Test that they have the expected attributes
        assert hasattr(user_aggregator, '_immediate_mode'), "User aggregator missing _immediate_mode"
        assert hasattr(assistant_aggregator, '_streaming_enabled'), "Assistant aggregator missing _streaming_enabled"
        
        print("✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating aggregators: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_compatibility():
    """Test that the aggregators work like standard pipeline aggregators."""
    print("\nTesting pipeline compatibility...")
    
    try:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        context = OpenAILLMContext(messages)
        
        # Create aggregators using our optimized version
        optimized_aggregator = create_gladia_optimized_aggregators(context)
        
        # Test that they behave like standard aggregators
        user_agg = optimized_aggregator.user()
        assistant_agg = optimized_aggregator.assistant()
        
        # Check that they have the basic methods we expect
        assert hasattr(user_agg, 'process_frame'), "User aggregator missing process_frame method"
        assert hasattr(assistant_agg, 'process_frame'), "Assistant aggregator missing process_frame method"
        assert hasattr(user_agg, 'reset'), "User aggregator missing reset method"
        assert hasattr(assistant_agg, 'reset'), "Assistant aggregator missing reset method"
        
        print("✅ Pipeline compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Gladia Optimized Aggregators\n")
    
    success1 = test_aggregator_creation()
    success2 = test_pipeline_compatibility()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The optimized aggregators should work correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the errors above.")
        sys.exit(1)
