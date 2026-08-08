#!/usr/bin/env python3
"""
Quick Performance Test for Seed-OSS-36B Deployments
Simple synchronous test with vLLM metrics integration
"""

import os
import time
import json
import requests
import statistics
from datetime import datetime

# TLS verification: enabled by default. For lab clusters with self-signed
# certificates either set REQUESTS_CA_BUNDLE to your cluster CA (preferred)
# or set PERF_TLS_VERIFY=false explicitly.
TLS_VERIFY = os.environ.get('PERF_TLS_VERIFY', 'true').strip().lower() not in ('0', 'false', 'no')

def test_endpoint(endpoint, api_key, test_query):
    """Test a single endpoint with timing measurements"""
    print(f"Testing {endpoint.split('//')[1].split('.')[0]}...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "ByteDance-Seed/Seed-OSS-36B-Instruct",
        "messages": [{"role": "user", "content": test_query}],
        "temperature": 0.3,
        "max_tokens": 100
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            headers=headers,
            json=payload,
            verify=TLS_VERIFY,  # See TLS_VERIFY note at top of file
            timeout=60
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            usage = data.get('usage', {})
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            total_tokens = usage.get('total_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
            
            return {
                "success": True,
                "total_time": total_time,
                "total_tokens": total_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_second": tokens_per_second,
                "response_length": len(content),
                "endpoint": endpoint
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "endpoint": endpoint
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "endpoint": endpoint
        }

def get_api_key():
    """Get API key from environment or OpenShift"""
    import subprocess
    try:
        result = subprocess.run([
            'oc', 'get', 'secret', 'seed-oss-api-key', 
            '-n', 'tme-aix', 
            '-o', 'jsonpath={.data.api-key}'
        ], capture_output=True, text=True, check=True)
        
        import base64
        return base64.b64decode(result.stdout).decode('utf-8')
    except Exception as e:
        print(f"Error getting API key: {e}")
        return None

def main():
    print("🎯 Quick Performance Test - Seed-OSS-36B")
    print("=" * 50)
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("❌ Failed to get API key")
        return
    
    # Test queries
    test_queries = [
        "What is 5G network slicing?",
        "Explain the benefits of edge computing for telecom operators.",
        "How do you troubleshoot network connectivity issues?"
    ]
    
    endpoints = {
        "Optimized": "https://seed-oss-36b-tme-aix.apps.sandbox02.narlabs.io"
    }
    
    results = {}
    
    for name, endpoint in endpoints.items():
        print(f"\n🚀 Testing {name} endpoint...")
        endpoint_results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"  Query {i}/3: {query[:30]}...")
            result = test_endpoint(endpoint, api_key, query)
            endpoint_results.append(result)
            
            if result['success']:
                print(f"    ✅ {result['tokens_per_second']:.1f} tokens/s in {result['total_time']:.2f}s")
            else:
                print(f"    ❌ Error: {result['error'][:50]}")
            
            time.sleep(2)  # Brief pause between requests
        
        results[name] = endpoint_results
    
    # Generate comparison report
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON REPORT")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Generated: {timestamp}")
    
    for name, endpoint_results in results.items():
        successful = [r for r in endpoint_results if r['success']]
        
        print(f"\n🔹 {name} Deployment:")
        print(f"   Success Rate: {len(successful)}/{len(endpoint_results)} ({len(successful)/len(endpoint_results)*100:.1f}%)")
        
        if successful:
            avg_tokens_per_sec = statistics.mean([r['tokens_per_second'] for r in successful])
            avg_total_time = statistics.mean([r['total_time'] for r in successful])
            total_tokens = sum([r['completion_tokens'] for r in successful])
            
            print(f"   Average Tokens/Second: {avg_tokens_per_sec:.2f}")
            print(f"   Average Response Time: {avg_total_time:.2f}s")
            print(f"   Total Tokens Generated: {total_tokens}")
        else:
            print("   No successful requests")
    
    # Performance analysis
    if results.get('Optimized'):
        optimized_successful = [r for r in results['Optimized'] if r['success']]
        
        if optimized_successful:
            avg_tps = statistics.mean([r['tokens_per_second'] for r in optimized_successful])
            avg_time = statistics.mean([r['total_time'] for r in optimized_successful])
            
            print(f"\n🚀 Performance Analysis:")
            print(f"   Average Performance: {avg_tps:.1f} tokens/s")
            print(f"   Average Response Time: {avg_time:.2f}s")
            
            if avg_tps > 25:
                print("   ✅ Excellent performance - exceeding target!")
            elif avg_tps > 20:
                print("   🔄 Good performance - meeting expectations")
            else:
                print("   ⚠️  Performance below optimal - check system load")
    
    # Save results
    report_file = f"quick_perf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {report_file}")
    print("\n🏁 Quick performance test complete!")

if __name__ == "__main__":
    main()