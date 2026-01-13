#include "Assets/bundled.h"
#include "Labs/Core/App.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace {
    // 从环境变量VCX_LOG_LEVEL配置spdlog日志级别
void ConfigureLoggingFromEnv() {
    const char* env = std::getenv("VCX_LOG_LEVEL");
    if (!env) {
        spdlog::set_level(spdlog::level::off);
        return;
    }
    std::string level{env};
    std::transform(level.begin(), level.end(), level.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (level == "off") {
        spdlog::set_level(spdlog::level::off);
    } else if (level == "critical") {
        spdlog::set_level(spdlog::level::critical);
    } else if (level == "error" || level == "err") {
        spdlog::set_level(spdlog::level::err);
    } else if (level == "warn" || level == "warning") {
        spdlog::set_level(spdlog::level::warn);
    } else if (level == "info") {
        spdlog::set_level(spdlog::level::info);
    } else if (level == "debug") {
        spdlog::set_level(spdlog::level::debug);
    } else if (level == "trace") {
        spdlog::set_level(spdlog::level::trace);
    }
}
} // namespace

int main() {
    using namespace VCX;

    ConfigureLoggingFromEnv();

    return Engine::RunApp<Labs::NeuralPhysicsSubspaces::App>(Engine::AppContextOptions {
        .Title      = "Neural Physics Subspaces",
        .WindowSize = { 1024, 768 },
        .FontSize   = 16,
        .IconFileNames = Assets::DefaultIcons,
        .FontFileNames = Assets::DefaultFonts,
    });
}
