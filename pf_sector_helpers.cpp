// g++ -fPIC -pthread -shared -o /problem/build/pf_sector_helpers.so
// /problem/pf_sector_helpers.cpp -std=c++11

#include <cstdint>

#include <iostream>

#include <memory>

#include <thread>

typedef double float64_t;

/**

* Looks at the input pf_num argument and writes the sector weights of that
portfolio to the portfolio_sector_weights argument.

* No size checking is done on the input pointers, so use this function with
great caution. See argument descriptions

* for input pointer structure.

*

* @param  pf_num which portfolio to calculate, used to move to relevant section
of memory in below pointers.

* @param  portfolio_bond_weights  pointer of size num_portfolios * num_bonds,
flattened. Each entry is the market value percent of a bond.

* @param  portfolio_bond_sectors pointer of size num_portfolios * num_bonds,
flattened. Each entry is the sector of a bond in integer representation.

* @param  num_bonds how many bonds per portfolio, assumed to be the same for
each portfolio. Used to move to relevant memory section.

* @param  portfolio_sector_weights the output array to write portfolio sector
weights to. Size of num_portfolios * num_sectors.

* @param  num_sectors the number of sectors in the sector scheme. Used to move
to relevant section of memory in portfolio_sector_weights.

* @return      void

* @see         calculate_sector_weights

*/

void worker_calculate_sector_weights(const uintmax_t pf_num,
                                     const float64_t *portfolio_bond_weights,
                                     const int32_t *portfolio_bond_sectors,

                                     const uintmax_t num_bonds,

                                     float64_t *portfolio_sector_weights,
                                     const uintmax_t num_sectors)

{

  const uintmax_t starting_out_index = pf_num * num_sectors;

  for (uintmax_t i = 0; i < num_sectors; i++)

  {

    portfolio_sector_weights[starting_out_index + i] = 0.0;
  }

  const uintmax_t starting_bond_index = pf_num * num_bonds;

  for (uintmax_t bond_index = starting_bond_index;
       bond_index < starting_bond_index + num_bonds; bond_index++)

  {

    const int32_t §or = portfolio_bond_sectors[bond_index];

    const float64_t &market_value_pct = portfolio_bond_weights[bond_index];

    portfolio_sector_weights[starting_out_index + sector] += market_value_pct;
  }
}

extern "C"

{

/**

 * Batches out calculations of sector weights in portfolios to threads. Uses a
 max of three threads at a time. Writes the sector weights to the

 *  portfolio_sector_weights argument. No size checking is done on the input
 pointers, so use this function with great caution.

 *  See argument descriptions for input pointer structure.

 *

 * @param  portfolio_bond_weights  pointer of size num_portfolios * num_bonds,
 flattened. Each entry is the market value percent of a bond.

 * @param  pf_size how many portfolios in input pointers, used to move to
 relevant section of memory in below pointers.

 * @param  portfolio_bond_sectors pointer of size num_portfolios * num_bonds,
 flattened. Each entry is the sector of a bond in integer representation.

 * @param  num_bonds how many bonds per portfolio, assumed to be the same for
 each portfolio. Used to move to relevant memory section.

 * @param  portfolio_sector_weights the output array to write portfolio sector
 weights to. Size of num_portfolios * num_sectors.

 * @param  num_sectors the number of sectors in the sector scheme. Used to move
 to relevant section of memory in portfolio_sector_weights.

 * @return      void

 */

void calculate_sector_weights(const float64_t *portfolios_bond_weights,
                              const uintmax_t pf_size,

                              const int32_t *portfolios_bond_sectors,
                              const uintmax_t num_bonds,

                              float64_t *portfolios_sector_weights,
                              const uintmax_t num_sectors)

{

  const uintmax_t num_workers = 3;

  std::thread thread_arr[num_workers];

  for (uintmax_t i = 0; i < pf_size; i++)

  {

    if (i % num_workers != 0)

    {

      continue;
    }

    for (uintmax_t j = 0; j < num_workers; j++)

    {

      uintmax_t pf_num = i + j;

      if (pf_num >= pf_size)

      {

        continue;
      }

      thread_arr[j] =
          std::thread(worker_calculate_sector_weights, pf_num,
                      portfolios_bond_weights, portfolios_bond_sectors,

                      num_bonds, portfolios_sector_weights, num_sectors);
    }

    for (uintmax_t j = 0; j < num_workers; j++)

    {

      if (i + j >= pf_size)

      {

        continue;
      }

      thread_arr[j].join();
    }
  }
}
}

int main()

{

  const uintmax_t num_bonds = 5000000;

  const uintmax_t num_sectors = 18;

  const uintmax_t num_portfolios = 20;

  const uintmax_t num_array_items = num_portfolios * num_bonds;

  float64_t *bond_weights_raw_ptr = new float64_t[num_array_items];

  std::unique_ptr<float64_t> bond_weights_ptr(bond_weights_raw_ptr);

  int32_t *bond_sectors_raw_ptr = new int32_t[num_array_items];

  std::unique_ptr<int32_t> bond_sectors_ptr(bond_sectors_raw_ptr);

  for (uintmax_t i = 0; i < num_array_items; i++)

  {

    bond_weights_ptr.get()[i] = 0.0001;

    bond_sectors_ptr.get()[i] = 1;
  }

  float64_t *out_raw_ptr = new float64_t[num_portfolios * num_sectors];

  std::unique_ptr<float64_t> sector_weights_ptr(out_raw_ptr);

  for (uintmax_t i = 0; i < num_portfolios * num_sectors; i++)

  {

    sector_weights_ptr.get()[i] = 0;
  }

  calculate_sector_weights(bond_weights_ptr.get(), num_portfolios,

                           bond_sectors_ptr.get(), num_bonds,

                           sector_weights_ptr.get(), num_sectors);
}