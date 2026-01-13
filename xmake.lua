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
    
    -- 支持 Conda 和系统 Python 环境
    on_load(function (target)
        local conda_prefix = os.getenv("CONDA_PREFIX")
        local python_path = nil
        local python_version = nil
        local python_version_short = nil
        
        -- 辅助函数：检测 Python 版本
        local function detect_python_version(python_exe)
            local version_output = os.iorun(python_exe .. " --version 2>&1")
            if version_output then
                -- 匹配 Python 3.x 或 Python 3.x.y
                local major, minor = version_output:match("Python (%d+)%.(%d+)")
                if major and minor then
                    return major .. "." .. minor, major .. minor
                end
            end
            return nil, nil
        end
        
        if conda_prefix then
            -- 优先使用 Conda 环境
            local python_exe = path.join(conda_prefix, is_host("windows") and "python.exe" or "bin/python")
            python_version, python_version_short = detect_python_version(python_exe)
            
            if python_version then
                print("Using Conda Python: " .. conda_prefix .. " (Python " .. python_version .. ")")
                python_path = conda_prefix
            else
                print("Warning: Could not detect Python version in Conda environment")
            end
        else
            -- 尝试从系统查找 Python
            print("CONDA_PREFIX not set, attempting to use system Python...")
            
            if is_host("windows") then
                -- Windows: 尝试从注册表或环境变量获取 Python 路径
                local python_exe = os.getenv("PYTHON_HOME")
                if not python_exe then
                    -- 尝试使用 where python 命令
                    local result = os.iorun("where python")
                    if result then
                        python_exe = result:trim()
                    end
                end
                
                if python_exe then
                    python_version, python_version_short = detect_python_version(python_exe)
                    if python_version then
                        python_path = path.directory(python_exe)
                        print("Found Python at: " .. python_path .. " (Python " .. python_version .. ")")
                    end
                end
            else
                -- Linux/macOS: 尝试从 which python3 获取
                local python_exe = os.iorun("which python3")
                if python_exe then
                    python_exe = python_exe:trim()
                    python_version, python_version_short = detect_python_version(python_exe)
                    if python_version then
                        python_path = path.directory(path.directory(python_exe))  -- 去掉 /bin/python3 得到根目录
                        print("Found Python at: " .. python_path .. " (Python " .. python_version .. ")")
                    end
                end
            end
        end
        
        if python_path and python_version then
            if is_host("windows") then
                target:add("includedirs", path.join(python_path, "include"), { public = true })
                target:add("linkdirs", path.join(python_path, "libs"), { public = true })
                target:add("links", "python" .. python_version_short, { public = true })
            else
                -- Linux/macOS: 动态查找 Python include 目录
                local include_dir = path.join(python_path, "include/python" .. python_version)
                if not os.isdir(include_dir) then
                    -- 有些系统可能使用 python3.xm 这样的目录名
                    include_dir = path.join(python_path, "include/python" .. python_version .. "m")
                end
                
                target:add("includedirs", include_dir, { public = true })
                target:add("linkdirs", path.join(python_path, "lib"), { public = true })
                
                -- 动态查找 libpython 库文件
                local lib_name = "python" .. python_version
                target:add("links", lib_name, { public = true })
                
                -- Set RPATH for Linux/macOS to find Python shared library at runtime
                target:add("ldflags", "-Wl,-rpath," .. path.join(python_path, "lib"), { force = true, public = true })
                
                print("Python configuration: include=" .. include_dir .. ", lib=" .. lib_name)
            end
        else
            print("Warning: Could not find Python. Build may fail.")
        end
    end)
    
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
    
    on_run(function (target)
        import("core.base.option")
        local args = option.get("arguments") or {}
        
        -- 支持 Conda 和系统 Python
        local conda_prefix = os.getenv("CONDA_PREFIX")
        local env = {}
        if conda_prefix then
            -- Conda 环境
            env["PYTHONHOME"] = conda_prefix
            if is_host("windows") then
                env["PATH"] = conda_prefix .. "\\Library\\bin;" .. conda_prefix .. ";" .. (os.getenv("PATH") or "")
            else
                -- Linux/macOS: Set LD_LIBRARY_PATH for conda environment
                env["PATH"] = path.join(conda_prefix, "bin") .. ":" .. (os.getenv("PATH") or "")
                env["LD_LIBRARY_PATH"] = path.join(conda_prefix, "lib") .. ":" .. (os.getenv("LD_LIBRARY_PATH") or "")
            end
        else
            -- 系统 Python 环境，只设置必要的 PATH
            local system_path = os.getenv("PATH") or ""
            env["PATH"] = system_path
        end
        
        os.execv(target:targetfile(), args, {curdir = target:rundir(), envs = env})
    end)