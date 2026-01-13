set_project("VCX-Labs")
set_version("2.0.0")
set_xmakever("2.6.9")
set_languages("cxx20")

add_rules("mode.debug", "mode.release", "mode.profile")

if is_plat("windows") then
    add_cxxflags("/utf-8")
end

add_requires("glad")
add_requires("glfw")
add_requires("glm 1.0.0")
add_requires("imgui 1.90.1")
add_requires("spdlog")
add_requires("stb")
add_requires("fmt")
add_requires("tinyobjloader")
add_requires("yaml-cpp")
add_requires("eigen")
add_requires("pybind11")


if is_plat("macosx") then
    add_defines("PLATFORM_MACOSX")
end

target("assets")
    set_kind("phony")
    set_default(true)
    after_build(function (target)
        os.mkdir(path.join(target:targetdir(), "assets"))
        os.cp("assets/*|shaders", path.join(target:targetdir(), "assets"))
        os.mkdir(path.join(target:targetdir(), "assets", "shaders"))
        os.cp("assets/shaders/*", path.join(target:targetdir(), "assets", "shaders"))
    end)
    after_install(function (target)
        os.mkdir(path.join(target:installdir(), "assets"))
        os.cp("assets/*|shaders", path.join(target:installdir(), "assets"))
        os.mkdir(path.join(target:installdir(), "assets", "shaders"))
        os.cp("assets/shaders/*", path.join(target:installdir(), "assets", "shaders"))
    end)
    after_clean(function (target)
        os.rm(path.join(target:targetdir(), "assets"))
    end)

target("engine")
    set_kind("static")
    add_rules("c++.unity_build", {batchsize = 0})
    add_packages("glad"         , { public = true })
    add_packages("glfw"         , { public = true })
    add_packages("glm"          , { public = true })
    add_packages("imgui"        , { public = true })
    add_packages("spdlog"       , { public = true })
    add_packages("stb"          , { public = true })
    add_packages("fmt"          , { public = true })
    add_packages("tinyobjloader", { public = true })
    add_packages("yaml-cpp"     , { public = true })
    add_packages("pybind11"     , { public = true })
    
    -- Python configuration: dynamically detect python path instead of hardcoding
    on_load(function (target)
        local python_prefix = os.getenv("CONDA_PREFIX")
        local python_ver = "3.9"
        
        if not python_prefix then
            -- Fallback to current python3 in path
            local result = os.iorun("python3 -c 'import sys; print(sys.prefix)'")
            if result then
                python_prefix = result:trim()
            end
        end

        if python_prefix then
            -- Detect version
            local result = os.iorun("python3 -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")'")
            if result then
                python_ver = result:trim()
            end

            print("Using Python environment: " .. python_prefix .. " (version " .. python_ver .. ")")
            target:add("includedirs", path.join(python_prefix, "include/python" .. python_ver), { public = true })
            target:add("linkdirs", path.join(python_prefix, "lib"), { public = true })
            target:add("rpathdirs", path.join(python_prefix, "lib"))
            target:add("links", "python" .. python_ver, { public = true })
        else
            print("Warning: Could not detect Python environment. Please set CONDA_PREFIX or ensure python3 is in PATH.")
        end
    end)

    add_includedirs("src/3rdparty", { public = true })
    add_includedirs("src/VCX"     , { public = true })
    add_headerfiles("src/3rdparty/**.h")
    add_headerfiles("src/3rdparty/**.hpp")
    add_files      ("src/3rdparty/**.cpp")
    add_headerfiles("src/VCX/Assets/**.h")
    add_headerfiles("src/VCX/Engine/**.h")
    add_headerfiles("src/VCX/Engine/**.hpp")
    add_files      ("src/VCX/Engine/**.cpp")

target("lab-common")
    set_kind("static")
    add_deps("engine")
    add_deps("assets")
    add_headerfiles("src/VCX/Labs/Common/*.h")
    add_files      ("src/VCX/Labs/Common/*.cpp")

target("NeuralPhysicsSubspaces")
    set_kind("binary")
    add_rules("c++.unity_build", {batchsize = 0})
    add_deps("lab-common")
    add_packages("eigen")
    set_rundir(os.projectdir()) 
    add_headerfiles("src/VCX/Labs/Core/*.h")
    add_files      ("src/VCX/Labs/Core/*.cpp")