//---------------------------------------------------------------------------//
// Copyright (c) 2013 Kyle Lutz <kyle.r.lutz@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//
// See http://boostorg.github.com/compute for more information.
//---------------------------------------------------------------------------//

#define BOOST_TEST_MODULE TestInplaceReduce
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <boost/compute/system.hpp>
#include <boost/compute/functional.hpp>
#include <boost/compute/algorithm/iota.hpp>
#include <boost/compute/algorithm/detail/inplace_reduce.hpp>
#include <boost/compute/container/vector.hpp>

#include "quirks.hpp"
#include "context_setup.hpp"

BOOST_AUTO_TEST_CASE(sum_int)
{
    if(is_apple_cpu_device(device)) {
        std::cerr
            << "skipping all inplace_reduce tests due to Apple platform"
            << " behavior when local memory is used on a CPU device"
            << std::endl;
        return;
    }

    int data[] = { 1, 5, 3, 4, 9, 3, 5, 3 };
    boost::compute::vector<int> vector(data, data + 8, queue);

    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(33));

    vector.assign(data, data + 8);
    vector.push_back(3);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(36));
}

BOOST_AUTO_TEST_CASE(multiply_int)
{
    if(is_apple_cpu_device(device)) {
        return;
    }

    int data[] = { 1, 5, 3, 4, 9, 3, 5, 3 };
    boost::compute::vector<int> vector(data, data + 8, queue);

    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::multiplies<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(24300));

    vector.assign(data, data + 8);
    vector.push_back(3);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::multiplies<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(72900));
}

BOOST_AUTO_TEST_CASE(reduce_iota)
{
    if(is_apple_cpu_device(device)) {
        return;
    }

    // 1 value
    boost::compute::vector<int> vector(1, context);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(0));

    // TEST: 12 values
    vector.resize(12);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(11 * 12 / 2));

    // TEST: 16 values
    vector.resize(16);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(15 * 16 / 2));

    // TEST: 48 values
    vector.resize(48);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(47 * 48 / 2));


    // TEST: 64 values
    vector.resize(64);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(63 * 64 / 2));

    // TEST: 66 values
    vector.resize(66);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(65 * 66 / 2));

    // TEST: 72 values
    vector.resize(72);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(71 * 72 / 2));

    // TEST: 80 values
    vector.resize(80);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(79 * 80 / 2));

    // TEST: 88 values
    vector.resize(88);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(87 * 88 / 2));

    // TEST: 90 values
    vector.resize(90);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(89 * 90 / 2));

    // TEST: 92 values
    vector.resize(92);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(91 * 92 / 2));


    // TEST: 94 values
    vector.resize(94);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(93 * 94 / 2));

    // TEST: 96 values
    vector.resize(96);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(95 * 96 / 2));


    // TEST: 100 values
    vector.resize(100);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(99 * 100 / 2));

    // TEST: 102 values
    vector.resize(102);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(101 * 102 / 2));

    // TEST: 104 values
    vector.resize(104);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(103 * 104 / 2));

    // TEST: 106 values
    vector.resize(106);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(105 * 106 / 2));

    // TEST: 120 values
    vector.resize(120);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(119 * 120 / 2));

    // TEST: 128 values
    vector.resize(128);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(127 * 128 / 2));

    // TEST: 130 values
    vector.resize(130);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(129 * 130 / 2));

    // TEST: 152 values
    vector.resize(152);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(151 * 152 / 2));

    // TEST: 160 values
    vector.resize(160);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(159 * 160 / 2));

    // TEST: 510 values
    vector.resize(510);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(509 * 510 / 2));

    // TEST: 512 values
    vector.resize(512);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(511 * 512 / 2));

    // TEST: 514 values
    vector.resize(514);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(513 * 514 / 2));

    // TEST: 574 values
    vector.resize(574);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(573 * 574 / 2));


    // 1000 values
    vector.resize(1000);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(499500));

    // 2499 values
    vector.resize(2499);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(3121251));

    // 4096 values
    vector.resize(4096);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(8386560));

    // TEST: 4094 values
    vector.resize(4094);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(4093 * 4094 / 2));

    // TEST: 4098 values
    vector.resize(4098);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(4097 * 4098 / 2));

    // 5000 values
    vector.resize(5000);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(12497500));

    // TEST: 5004 values
    vector.resize(5004);
    boost::compute::iota(vector.begin(), vector.end(), int(0), queue);
    boost::compute::detail::inplace_reduce(vector.begin(),
                                           vector.end(),
                                           boost::compute::plus<int>(),
                                           queue);
    queue.finish();
    BOOST_CHECK_EQUAL(int(vector[0]), int(5003 * 5004 / 2));
}

BOOST_AUTO_TEST_SUITE_END()
