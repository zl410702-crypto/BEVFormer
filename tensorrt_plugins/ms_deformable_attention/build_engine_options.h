#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

struct BuilderOptions
{
    std::string model;
    std::string plugin;
    std::string engine;
    uint64_t workspaceMiB{4096};
    bool reducedTacticSources{false};
    int32_t minTimingIterations{-1};
    int32_t avgTimingIterations{-1};
    int32_t expectedPluginLayers{12};
};

inline BuilderOptions parseBuilderOptions(int argc, char** argv)
{
    if (argc < 4)
    {
        throw std::invalid_argument("three positional arguments are required");
    }
    BuilderOptions options{argv[1], argv[2], argv[3]};
    for (int index = 4; index < argc; ++index)
    {
        std::string const argument{argv[index]};
        std::string const workspacePrefix{"--workspace-mib="};
        std::string const tacticPrefix{"--tactic-sources="};
        std::string const minTimingPrefix{"--min-timing-iterations="};
        std::string const avgTimingPrefix{"--avg-timing-iterations="};
        std::string const expectedPluginPrefix{"--expected-plugin-layers="};
        if (argument.compare(0, workspacePrefix.size(), workspacePrefix) == 0)
        {
            std::string const value = argument.substr(workspacePrefix.size());
            std::size_t parsed = 0;
            options.workspaceMiB = std::stoull(value, &parsed);
            if (parsed != value.size() || options.workspaceMiB == 0 ||
                options.workspaceMiB > 65536)
            {
                throw std::invalid_argument("workspace MiB must be in [1,65536]");
            }
        }
        else if (argument.compare(0, tacticPrefix.size(), tacticPrefix) == 0)
        {
            std::string const value = argument.substr(tacticPrefix.size());
            if (value == "default")
            {
                options.reducedTacticSources = false;
            }
            else if (value == "reduced")
            {
                options.reducedTacticSources = true;
            }
            else
            {
                throw std::invalid_argument(
                    "tactic sources must be default or reduced");
            }
        }
        else if (argument.compare(0, minTimingPrefix.size(), minTimingPrefix) == 0)
        {
            std::string const value = argument.substr(minTimingPrefix.size());
            std::size_t parsed = 0;
            unsigned long const parsedValue = std::stoul(value, &parsed);
            if (parsed != value.size() || parsedValue == 0 || parsedValue > 100)
            {
                throw std::invalid_argument(
                    "minimum timing iterations must be in [1,100]");
            }
            options.minTimingIterations = static_cast<int32_t>(parsedValue);
        }
        else if (argument.compare(0, avgTimingPrefix.size(), avgTimingPrefix) == 0)
        {
            std::string const value = argument.substr(avgTimingPrefix.size());
            std::size_t parsed = 0;
            unsigned long const parsedValue = std::stoul(value, &parsed);
            if (parsed != value.size() || parsedValue == 0 || parsedValue > 100)
            {
                throw std::invalid_argument(
                    "average timing iterations must be in [1,100]");
            }
            options.avgTimingIterations = static_cast<int32_t>(parsedValue);
        }
        else if (argument.compare(0, expectedPluginPrefix.size(),
                                  expectedPluginPrefix) == 0)
        {
            std::string const value = argument.substr(expectedPluginPrefix.size());
            std::size_t parsed = 0;
            unsigned long const parsedValue = std::stoul(value, &parsed);
            if (parsed != value.size() || parsedValue > 1024)
            {
                throw std::invalid_argument(
                    "expected plugin layers must be in [0,1024]");
            }
            options.expectedPluginLayers = static_cast<int32_t>(parsedValue);
        }
        else
        {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return options;
}
