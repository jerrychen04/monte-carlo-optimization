#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <ctime>
#include <thread>
#include <mutex>
#include <functional>
#include <chrono>
using namespace std;

double monte_carlo_american_option_pricing_single(double init_price, double strike, double ttm, double int_rate, double volatility,
                                                  const string &type_of_option = "call", size_t num_simulations = 500000)
{
  random_device rd;
  mt19937 gen(rd());
  normal_distribution<> dist(0.0, 1.0);

  double dt = ttm / static_cast<double>(static_cast<size_t>(ttm * 365));
  vector<double> option_payoffs(num_simulations);

  for (size_t i = 0; i < num_simulations; ++i)
  {
    double current_price = init_price;
    double option_payoff = (type_of_option == "call") ? -INFINITY : INFINITY;

    for (size_t j = 0; j < static_cast<size_t>(ttm * 365); ++j)
    {
      current_price *= exp((int_rate - 0.5 * volatility * volatility) * dt + volatility * sqrt(dt) * dist(gen));

      if (type_of_option == "call")
      {
        option_payoff = max(option_payoff, current_price - strike);
      }
      else if (type_of_option == "put")
      {
        option_payoff = min(option_payoff, strike - current_price);
      }
      else
      {
        throw invalid_argument("Invalid option type. Use 'call' or 'put'.");
      }
    }

    option_payoffs[i] = exp(-int_rate * ttm) * option_payoff;
  }

  double simulated_priced_option = accumulate(option_payoffs.begin(), option_payoffs.end(), 0.0) / static_cast<double>(num_simulations);

  return simulated_priced_option;
}

void monte_carlo_american_option_pricing_worker(double init_price, double strike, double ttm, double int_rate, double volatility,
                                                const string &type_of_option,
                                                size_t start_index, size_t end_index, double &partial_sum)
{
  random_device rd;
  mt19937 gen(rd());
  normal_distribution<> dist(0.0, 1.0);
  double dt = ttm / static_cast<double>(static_cast<size_t>(ttm * 365));
  double local_partial_sum = 0.0;

  for (size_t i = start_index; i < end_index; ++i)
  {
    double current_price = init_price;
    double option_payoff = (type_of_option == "call") ? -INFINITY : INFINITY;

    for (size_t j = 0; j < static_cast<size_t>(ttm * 365); ++j)
    {
      current_price *= exp((int_rate - 0.5 * volatility * volatility) * dt + volatility * sqrt(dt) * dist(gen));

      if (type_of_option == "call")
      {
        option_payoff = max(option_payoff, current_price - strike);
      }
      else if (type_of_option == "put")
      {
        option_payoff = min(option_payoff, strike - current_price);
      }
      else
      {
        throw invalid_argument("Invalid option type. Use 'call' or 'put'.");
      }
    }

    local_partial_sum += exp(-int_rate * ttm) * option_payoff;
  }

  partial_sum = local_partial_sum;
}

double monte_carlo_american_option_pricing_multi(double init_price, double strike, double ttm, double int_rate, double volatility,
                                                 const string &type_of_option = "call", size_t num_simulations = 500000, size_t num_threads = thread::hardware_concurrency())
{
  vector<double> partial_sums(num_threads);
  vector<thread> threads(num_threads);

  size_t simulations_per_thread = num_simulations / num_threads;

  for (size_t i = 0; i < num_threads; ++i)
  {
    size_t start_index = i * simulations_per_thread;
    size_t end_index = (i == num_threads - 1) ? num_simulations : (i + 1) * simulations_per_thread;

    threads[i] = thread(monte_carlo_american_option_pricing_worker, init_price, strike, ttm, int_rate, volatility,
                        ref(type_of_option), start_index, end_index, ref(partial_sums[i]));
  }

  for (auto &t : threads)
  {
    t.join();
  }

  double total_payoff_sum = accumulate(partial_sums.begin(), partial_sums.end(), 0.0);
  double simulated_priced_option = total_payoff_sum / static_cast<double>(num_simulations);

  return simulated_priced_option;
}

int main()
{
  cout.precision(10);

  double init_price = 100.0;
  double strike = 110.0;
  double ttm = 1.0;
  double int_rate = 0.05;
  double volatility = 0.2;
  size_t num_simulations = 500000;

  auto start_time_single = chrono::high_resolution_clock::now();
  double call_option_price_single = monte_carlo_american_option_pricing_single(init_price, strike, ttm, int_rate, volatility, "call", num_simulations);
  auto end_time_single = chrono::high_resolution_clock::now();
  auto duration_single = chrono::duration_cast<chrono::milliseconds>(end_time_single - start_time_single).count();

  auto start_time_multi = chrono::high_resolution_clock::now();
  double call_option_price_multi = monte_carlo_american_option_pricing_multi(init_price, strike, ttm, int_rate, volatility, "call", num_simulations);
  auto end_time_multi = chrono::high_resolution_clock::now();
  auto duration_multi = chrono::duration_cast<chrono::milliseconds>(end_time_multi - start_time_multi).count();

  cout << "Single-threaded American call option price: " << call_option_price_single << endl;
  cout << "Single-threaded runtime: " << duration_single << " ms" << endl;

  cout << "Multithreaded American call option price: " << call_option_price_multi << endl;
  cout << "Multithreaded runtime: " << duration_multi << " ms" << endl;

  return 0;
}
