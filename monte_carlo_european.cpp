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

double monte_carlo_european_option_pricing_single(double init_price, double strike, double ttm, double int_rate, double volatility,
                                                  const string &type_of_option = "call", size_t num_simulations = 1000000)
{
    // Generate random numbers for the Monte Carlo simulation
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0.0, 1.0);

    vector<double> rand_nums(num_simulations * static_cast<size_t>(ttm * 365));
    for (auto &num : rand_nums)
    {
        num = dist(gen);
    }

    // Calculate the stock price at maturity for each simulation
    double dt = ttm / static_cast<double>(static_cast<size_t>(ttm * 365));
    vector<double> price_at_maturity(num_simulations);
    for (size_t i = 0; i < num_simulations; ++i)
    {
        double sum = 0;
        for (size_t j = 0; j < static_cast<size_t>(ttm * 365); ++j)
        {
            sum += (int_rate - 0.5 * volatility * volatility) * dt + volatility * sqrt(dt) * rand_nums[i * static_cast<size_t>(ttm * 365) + j];
        }
        price_at_maturity[i] = init_price * exp(sum);
    }

    // Determine the payoff for each simulation type (call or put)
    vector<double> payoff(num_simulations);
    if (type_of_option == "call")
    {
        for (size_t i = 0; i < num_simulations; ++i)
        {
            payoff[i] = max(price_at_maturity[i] - strike, 0.0);
        }
    }
    else if (type_of_option == "put")
    {
        for (size_t i = 0; i < num_simulations; ++i)
        {
            payoff[i] = max(strike - price_at_maturity[i], 0.0);
        }
    }
    else
    {
        throw invalid_argument("Invalid option type. Use 'call' or 'put'.");
    }

    // Calculate the present value of the payoffs using interest rate
    vector<double> present_value(num_simulations);
    for (size_t i = 0; i < num_simulations; ++i)
    {
        present_value[i] = exp(-int_rate * ttm) * payoff[i];
    }

    // Estimate the option price as the mean of the present values from all our simulations
    double simulated_priced_option = accumulate(present_value.begin(),
                                                present_value.end(), 0.0) /
                                     static_cast<double>(num_simulations);

    return simulated_priced_option;
}

// Declare a mutex to protect shared resources
mutex mtx;

void monte_carlo_european_option_pricing_worker(double init_price, double strike, double ttm, double int_rate, double volatility,
                                                const string &type_of_option,
                                                size_t start_index, size_t end_index, vector<double> &present_value)
{
    // Generate random numbers for the Monte Carlo simulation
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0.0, 1.0);

    double dt = ttm / static_cast<double>(static_cast<size_t>(ttm * 365));

    for (size_t i = start_index; i < end_index; ++i)
    {
        vector<double> rand_nums(static_cast<size_t>(ttm * 365));
        for (auto &num : rand_nums)
        {
            num = dist(gen);
        }

        double sum = 0;
        for (size_t j = 0; j < static_cast<size_t>(ttm * 365); ++j)
        {
            sum += (int_rate - 0.5 * volatility * volatility) * dt + volatility * sqrt(dt) * rand_nums[j];
        }

        double price_at_maturity = init_price * exp(sum);
        double payoff;

        if (type_of_option == "call")
        {
            payoff = max(price_at_maturity - strike, 0.0);
        }
        else if (type_of_option == "put")
        {
            payoff = max(strike - price_at_maturity, 0.0);
        }
        else
        {
            throw invalid_argument("Invalid option type. Use 'call' or 'put'.");
        }

        // Lock the mutex to protect shared resources
        unique_lock<mutex> lock(mtx);
        present_value[i] = exp(-int_rate * ttm) * payoff;
        lock.unlock();
    }
}

double monte_carlo_european_option_pricing_multi(double init_price, double strike, double ttm, double int_rate, double volatility,
                                                 const string &type_of_option = "call", size_t num_simulations = 1000000, size_t num_threads = thread::hardware_concurrency())
{
    vector<double> present_value(num_simulations);
    vector<thread> threads(num_threads);

    size_t simulations_per_thread = num_simulations / num_threads;

    for (size_t i = 0; i < num_threads; ++i)
    {
        size_t start_index = i * simulations_per_thread;
        size_t end_index = (i == num_threads - 1) ? num_simulations : (i + 1) * simulations_per_thread;

        threads[i] = thread(monte_carlo_european_option_pricing_worker, init_price, strike, ttm, int_rate, volatility,
                            ref(type_of_option), start_index, end_index, ref(present_value));
    }

    for (auto &t : threads)
    {
        t.join();
    }

    double simulated_priced_option = accumulate(present_value.begin(), present_value.end(), 0.0) / static_cast<double>(num_simulations);

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
    size_t num_simulations = 1000000;

    auto start_time_single = chrono::high_resolution_clock::now();
    double call_option_price_single = monte_carlo_european_option_pricing_single(init_price, strike, ttm, int_rate, volatility, "call", num_simulations);
    auto end_time_single = chrono::high_resolution_clock::now();
    auto duration_single = chrono::duration_cast<chrono::milliseconds>(end_time_single - start_time_single).count();

    auto start_time_multi = chrono::high_resolution_clock::now();
    double call_option_price_multi = monte_carlo_european_option_pricing_multi(init_price, strike, ttm, int_rate, volatility, "call", num_simulations);
    auto end_time_multi = chrono::high_resolution_clock::now();
    auto duration_multi = chrono::duration_cast<chrono::milliseconds>(end_time_multi - start_time_multi).count();

    cout << "Single-threaded call option price: " << call_option_price_single << endl;
    cout << "Single-threaded runtime: " << duration_single << " ms" << endl;

    cout << "Multithreaded call option price: " << call_option_price_multi << endl;
    cout << "Multithreaded runtime: " << duration_multi << " ms" << endl;

    return 0;
}