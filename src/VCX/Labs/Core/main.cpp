#include "Assets/bundled.h"
#include "Labs/Core/App.h"

int main() {
    using namespace VCX;
    return Engine::RunApp<Labs::NeuralPhysicsSubspaces::App>(Engine::AppContextOptions {
        .Title      = "Neural Physics Subspaces",
        .WindowSize = { 1024, 768 },
        .FontSize   = 16,
        .IconFileNames = Assets::DefaultIcons,
        .FontFileNames = Assets::DefaultFonts,
    });
}
