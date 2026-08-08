#!/usr/bin/env python3
"""
Performance Test Suite for Seed-OSS-36B Deployments
Compares standard vs high-performance vLLM deployments with telecom-specific test cases
"""

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any
import argparse

class PerformanceTestSuite:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.standard_endpoint = "https://seed-oss-36b-tme-aix.apps.sandbox02.narlabs.io"
        self.hp_endpoint = "https://seed-oss-36b-hp-tme-aix.apps.sandbox02.narlabs.io"
        self.model_name = "ByteDance-Seed/Seed-OSS-36B-Instruct"
        
        # Telecom-specific test queries
        self.test_queries = [
            {
                "name": "5G Network Optimization",
                "query": "Explain how to optimize 5G network performance for massive IoT deployments in urban environments. Consider beam management, interference mitigation, and energy efficiency.",
                "max_tokens": 300,
                "category": "technical"
            },
            {
                "name": "Network Troubleshooting",
                "query": "A customer reports intermittent call drops in a specific cell tower area. What are the step-by-step troubleshooting procedures for identifying and resolving this issue?",
                "max_tokens": 250,
                "category": "operational"
            },
            {
                "name": "Edge Computing Analysis",
                "query": "Compare the benefits and challenges of implementing Multi-access Edge Computing (MEC) for telecom operators. Include latency, cost, and scalability considerations.",
                "max_tokens": 400,
                "category": "strategic"
            },
            {
                "name": "Network Slicing Configuration",
                "query": "How do you configure network slicing for different service types: enhanced mobile broadband (eMBB), ultra-reliable low-latency communications (URLLC), and massive machine-type communications (mMTC)?",
                "max_tokens": 350,
                "category": "technical"
            },
            {
                "name": "Spectrum Management",
                "query": "What are the key considerations for dynamic spectrum sharing between 4G and 5G networks? Explain interference management techniques.",
                "max_tokens": 200,
                "category": "technical"
            },
            {
                "name": "Customer Service Response",
                "query": "A business customer is experiencing slow data speeds during peak hours. Provide a professional response explaining possible causes and resolution timeline.",
                "max_tokens": 150,
                "category": "customer_service"
            },
            {
                "name": "Network Security Assessment",
                "query": "Describe the security vulnerabilities in 5G standalone (SA) networks and recommended mitigation strategies for telecom operators.",
                "max_tokens": 300,
                "category": "security"
            },
            {
                "name": "Infrastructure Planning",
                "query": "Calculate the optimal cell tower density for providing 5G coverage in a suburban area with 50,000 residents. Consider terrain, building density, and service requirements.",
                "max_tokens": 250,
                "category": "planning"
            }
        ]

    async def make_request(self, session: aiohttp.ClientSession, endpoint: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single API request and measure performance metrics"""
        start_time = time.time()
        first_token_time = None
        tokens = []
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": query["query"]
                }
            ],
            "temperature": 0.3,
            "max_tokens": query["max_tokens"],
            "stream": True
        }
        
        try:
            async with session.post(
                f"{endpoint}/v1/chat/completions",
                headers=headers,
                json=payload,
                ssl=False,  # Ignore SSL for OpenShift self-signed certs
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status != 200:
                    return {
                        "error": f"HTTP {response.status}",
                        "query_name": query["name"]
                    }
                
                response_text = ""
                token_times = []
                
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str != '[DONE]':
                                try:
                                    chunk = json.loads(data_str)
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            current_time = time.time()
                                            if first_token_time is None:
                                                first_token_time = current_time
                                            token_times.append(current_time)
                                            response_text += content
                                except json.JSONDecodeError:
                                    continue
                
                end_time = time.time()
                total_time = end_time - start_time
                time_to_first_token = (first_token_time - start_time) if first_token_time else 0
                
                # Calculate token metrics
                total_tokens = len(response_text.split())  # Rough token estimate
                tokens_per_second = total_tokens / total_time if total_time > 0 else 0
                
                # Calculate inter-token latency
                inter_token_latencies = []
                if len(token_times) > 1:
                    for i in range(1, len(token_times)):
                        inter_token_latencies.append(token_times[i] - token_times[i-1])
                
                return {
                    "query_name": query["name"],
                    "category": query["category"],
                    "success": True,
                    "total_time": total_time,
                    "time_to_first_token": time_to_first_token,
                    "tokens_generated": total_tokens,
                    "tokens_per_second": tokens_per_second,
                    "inter_token_latency_avg": statistics.mean(inter_token_latencies) if inter_token_latencies else 0,
                    "inter_token_latency_p95": statistics.quantiles(inter_token_latencies, n=20)[18] if len(inter_token_latencies) > 10 else 0,
                    "response_length": len(response_text),
                    "endpoint": endpoint
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "query_name": query["name"],
                "endpoint": endpoint,
                "success": False
            }

    async def run_benchmark(self, endpoint: str, runs: int = 3) -> List[Dict[str, Any]]:
        """Run benchmark tests on a single endpoint"""
        print(f"\n🚀 Testing {endpoint.split('//')[1].split('.')[0]} endpoint...")
        
        results = []
        async with aiohttp.ClientSession() as session:
            for run in range(runs):
                print(f"  Run {run + 1}/{runs}")
                for i, query in enumerate(self.test_queries):
                    print(f"    Query {i + 1}/{len(self.test_queries)}: {query['name'][:30]}...")
                    result = await self.make_request(session, endpoint, query)
                    result["run"] = run + 1
                    results.append(result)
                    
                    # Small delay between requests
                    await asyncio.sleep(1)
                
                # Longer delay between runs
                if run < runs - 1:
                    print(f"    Waiting 10s before next run...")
                    await asyncio.sleep(10)
        
        return results

    def analyze_results(self, standard_results: List[Dict], hp_results: List[Dict]) -> Dict[str, Any]:
        """Analyze and compare results from both endpoints"""
        
        def aggregate_metrics(results):
            successful_results = [r for r in results if r.get('success', False)]
            if not successful_results:
                return {}
            
            return {
                "total_requests": len(results),
                "successful_requests": len(successful_results),
                "success_rate": len(successful_results) / len(results) * 100,
                "avg_total_time": statistics.mean([r['total_time'] for r in successful_results]),
                "avg_time_to_first_token": statistics.mean([r['time_to_first_token'] for r in successful_results]),
                "avg_tokens_per_second": statistics.mean([r['tokens_per_second'] for r in successful_results]),
                "avg_inter_token_latency": statistics.mean([r['inter_token_latency_avg'] for r in successful_results if r['inter_token_latency_avg'] > 0]),
                "p95_inter_token_latency": statistics.mean([r['inter_token_latency_p95'] for r in successful_results if r['inter_token_latency_p95'] > 0]),
                "total_tokens_generated": sum([r['tokens_generated'] for r in successful_results]),
                "avg_response_length": statistics.mean([r['response_length'] for r in successful_results])
            }
        
        standard_metrics = aggregate_metrics(standard_results)
        hp_metrics = aggregate_metrics(hp_results)
        
        # Calculate improvements
        improvements = {}
        if standard_metrics and hp_metrics:
            improvements = {
                "tokens_per_second_improvement": ((hp_metrics['avg_tokens_per_second'] - standard_metrics['avg_tokens_per_second']) / standard_metrics['avg_tokens_per_second'] * 100) if standard_metrics['avg_tokens_per_second'] > 0 else 0,
                "time_to_first_token_improvement": ((standard_metrics['avg_time_to_first_token'] - hp_metrics['avg_time_to_first_token']) / standard_metrics['avg_time_to_first_token'] * 100) if standard_metrics['avg_time_to_first_token'] > 0 else 0,
                "total_time_improvement": ((standard_metrics['avg_total_time'] - hp_metrics['avg_total_time']) / standard_metrics['avg_total_time'] * 100) if standard_metrics['avg_total_time'] > 0 else 0,
                "inter_token_latency_improvement": ((standard_metrics['avg_inter_token_latency'] - hp_metrics['avg_inter_token_latency']) / standard_metrics['avg_inter_token_latency'] * 100) if standard_metrics.get('avg_inter_token_latency', 0) > 0 else 0
            }
        
        return {
            "standard": standard_metrics,
            "high_performance": hp_metrics,
            "improvements": improvements
        }

    def generate_report(self, analysis: Dict[str, Any], output_file: str = None):
        """Generate detailed performance report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Seed-OSS-36B Performance Comparison Report
Generated: {timestamp}

## Test Configuration
- Model: {self.model_name}
- Test Queries: {len(self.test_queries)} telecom-specific scenarios
- Standard Endpoint: {self.standard_endpoint}
- High-Performance Endpoint: {self.hp_endpoint}

## Performance Metrics Comparison

### Standard Deployment Results
"""
        
        if analysis.get('standard'):
            std = analysis['standard']
            report += f"""
- **Success Rate**: {std['success_rate']:.1f}% ({std['successful_requests']}/{std['total_requests']})
- **Average Tokens/Second**: {std['avg_tokens_per_second']:.2f}
- **Average Time to First Token**: {std['avg_time_to_first_token']:.3f}s
- **Average Total Response Time**: {std['avg_total_time']:.2f}s
- **Average Inter-Token Latency**: {std.get('avg_inter_token_latency', 0):.4f}s
- **P95 Inter-Token Latency**: {std.get('p95_inter_token_latency', 0):.4f}s
- **Total Tokens Generated**: {std['total_tokens_generated']:,}
"""

        report += "\n### High-Performance Deployment Results\n"
        
        if analysis.get('high_performance'):
            hp = analysis['high_performance']
            report += f"""
- **Success Rate**: {hp['success_rate']:.1f}% ({hp['successful_requests']}/{hp['total_requests']})
- **Average Tokens/Second**: {hp['avg_tokens_per_second']:.2f}
- **Average Time to First Token**: {hp['avg_time_to_first_token']:.3f}s
- **Average Total Response Time**: {hp['avg_total_time']:.2f}s
- **Average Inter-Token Latency**: {hp.get('avg_inter_token_latency', 0):.4f}s
- **P95 Inter-Token Latency**: {hp.get('p95_inter_token_latency', 0):.4f}s
- **Total Tokens Generated**: {hp['total_tokens_generated']:,}
"""

        report += "\n### Performance Improvements (HP vs Standard)\n"
        
        if analysis.get('improvements'):
            imp = analysis['improvements']
            report += f"""
- **Tokens/Second**: {imp['tokens_per_second_improvement']:+.1f}%
- **Time to First Token**: {imp['time_to_first_token_improvement']:+.1f}%
- **Total Response Time**: {imp['total_time_improvement']:+.1f}%
- **Inter-Token Latency**: {imp['inter_token_latency_improvement']:+.1f}%
"""

        report += f"""

## Test Scenarios
The following telecom-specific scenarios were tested:

"""
        for i, query in enumerate(self.test_queries, 1):
            report += f"{i}. **{query['name']}** ({query['category']})\n"
            report += f"   - Max tokens: {query['max_tokens']}\n"
            report += f"   - Query: {query['query'][:100]}...\n\n"

        report += """
## Recommendations

Based on the performance comparison:

1. **For High-Throughput Workloads**: Use high-performance endpoint for batch processing and complex queries
2. **For Standard Operations**: Use standard endpoint for regular customer service and simple queries  
3. **Cost Optimization**: Route traffic based on complexity - simple queries to standard, complex to HP
4. **Monitoring**: Continue monitoring GPU utilization and adjust based on actual usage patterns

## Technical Notes

- SSL verification disabled for OpenShift self-signed certificates
- Streaming responses used for real-time metrics collection
- Token counting estimated using word splitting (approximate)
- Multiple runs performed to account for variability
- Inter-token latency measured using streaming response timing
"""

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\n📄 Report saved to: {output_file}")
        
        return report

async def main():
    parser = argparse.ArgumentParser(description="Performance test suite for Seed-OSS-36B deployments")
    parser.add_argument("--api-key", required=True, help="API authentication key")
    parser.add_argument("--runs", type=int, default=3, help="Number of test runs per endpoint")
    parser.add_argument("--output", help="Output file for report (default: print to stdout)")
    parser.add_argument("--endpoint", choices=['standard', 'hp', 'both'], default='both', 
                       help="Which endpoint(s) to test")
    
    args = parser.parse_args()
    
    suite = PerformanceTestSuite(args.api_key)
    
    print("🎯 Seed-OSS-36B Performance Test Suite")
    print("=" * 50)
    print(f"Model: {suite.model_name}")
    print(f"Test queries: {len(suite.test_queries)}")
    print(f"Runs per endpoint: {args.runs}")
    
    standard_results = []
    hp_results = []
    
    if args.endpoint in ['standard', 'both']:
        standard_results = await suite.run_benchmark(suite.standard_endpoint, args.runs)
    
    if args.endpoint in ['hp', 'both']:
        hp_results = await suite.run_benchmark(suite.hp_endpoint, args.runs)
    
    if standard_results and hp_results:
        analysis = suite.analyze_results(standard_results, hp_results)
        report = suite.generate_report(analysis, args.output)
        
        if not args.output:
            print(report)
    elif standard_results:
        print(f"\n✅ Standard endpoint tested: {len([r for r in standard_results if r.get('success')])}/{len(standard_results)} successful")
    elif hp_results:
        print(f"\n✅ High-performance endpoint tested: {len([r for r in hp_results if r.get('success')])}/{len(hp_results)} successful")
    
    print("\n🏁 Performance testing complete!")

if __name__ == "__main__":
    asyncio.run(main())