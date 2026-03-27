from OpenGL import GL

def create_and_compile_shader(stype, source):
    shader = GL.glCreateShader(stype)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)

    result = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)

    if result != 1:
        raise Exception("Couldn't compile shader\nShader compilation Log:\n" + str(GL.glGetShaderInfoLog(shader)))
    return shader

vertex_shader = None
vertex_wireframe_shader = None
vertex_tex_shader = None
fragment_shader = None
fragment_tex_shader = None
fragment_wireframe_shader = None
fragment_sh_shader = None
fragment_tex_sh_shader = None
fragment_tex_sh_shader_text = None
vertex_id_shader = None
fragment_id_shader = None
vertex_lines_shader = None
fragment_lines_shader = None
program = None
program_tex = None
program_tex_ao = None
program_sh = None
program_sh_tex = None
program_sh_tex_ao = None
program_wireframe = None
program_lines = None

def initialize():
    global vertex_shader
    global vertex_wireframe_shader
    global vertex_tex_shader
    global fragment_shader
    global fragment_tex_shader
    global fragment_wireframe_shader
    global fragment_sh_shader
    global fragment_tex_sh_shader
    global fragment_tex_sh_shader_text
    global vertex_id_shader
    global fragment_id_shader
    global vertex_lines_shader
    global fragment_lines_shader
    global id_program
    global program
    global program_tex
    global program_tex_ao
    global program_sh
    global program_sh_tex
    global program_sh_tex_ao
    global program_wireframe
    global program_lines

    vertex_shader=create_and_compile_shader(GL.GL_VERTEX_SHADER,"""
    #version 330 core

    layout(location = 0) in vec3 vertexPosition;
    layout(location = 1) in vec3 vertexNormal;
    layout(location = 2) in vec3 vertexColor;
    layout(location = 3) in float vertexVisible;

    out vec3 normal;
    out vec3 c;
    out vec3 position;
    out float visible;

    uniform mat4 MVP;

    void main(void)
    {
        position = vertexPosition;
        gl_Position = MVP * vec4(vertexPosition, 1);
        normal = vertexNormal;
        c = vertexColor;
        visible = vertexVisible;
    }
    """)

    vertex_wireframe_shader=create_and_compile_shader(GL.GL_VERTEX_SHADER,"""
    #version 330 core

    layout(location = 0) in vec3 vertexPosition;

    out vec3 c;

    uniform mat4 MVP;
    uniform vec3 amb; // Overloaded to use as wireframe color
    uniform float bias;

    void main(void)
    {
        gl_Position = MVP * vec4(vertexPosition, 1);
        gl_Position.z += bias;
        c = amb;
    }
    """)

    vertex_tex_shader=create_and_compile_shader(GL.GL_VERTEX_SHADER,"""
    #version 330 core

    layout(location = 0) in vec3 vertexPosition;
    layout(location = 1) in vec3 vertexNormal;
    layout(location = 2) in vec3 vertexColor;
    layout(location = 3) in float vertexVisible;
    layout(location = 4) in vec2 vertexUV;
    layout(location = 5) in vec3 vertexTan;
    layout(location = 6) in vec3 vertexBitan;

    out vec3 normal;
    out vec3 c;
    out vec3 position;
    out float visible;
    out vec2 uv;
    out vec3 tangent;
    out vec3 bitangent;

    uniform mat4 MVP;

    void main(void)
    {
        position = vertexPosition;
        gl_Position = MVP * vec4(vertexPosition, 1);
        normal = vertexNormal;
        c = vertexColor;
        visible = vertexVisible;
        uv = vec2(vertexUV.x, 1 - vertexUV.y);
        tangent = vertexTan;
        bitangent = vertexBitan;
    }
    """)

    fragment_shader=create_and_compile_shader(GL.GL_FRAGMENT_SHADER,"""
    #version 330 core

    in vec3 position;
    in vec3 normal;
    in vec3 c;
    in float visible;

    out vec4 color;

    uniform vec3 amb;
    uniform vec3 direct;
    uniform vec3 directDir;
    uniform vec3 camT;
    uniform float specAng;
    uniform float specMag;
    uniform float twoSided;

    void main(void)
    {
        vec3 n = normalize(normal);
        vec3 l = -normalize(directDir);
        float ref = max(0, dot(n, l)) + twoSided * max(0, dot(-n, l));
        vec3 d = direct * ref;
        vec3 tex = (amb + d) * c;

        // Compute shininess
        vec3 refDir = (2 * dot(n, l) * n - l);
        vec3 camDiff = camT - position;
        vec3 H = normalize((l + normalize(camDiff)));
        float refDiff = max(0, dot(refDir, H));
        vec3 shine = direct * specMag * pow(refDiff, specAng);

        color = vec4(visible * (tex + shine), 1.0);
    }
    """)

    fragment_tex_shader=create_and_compile_shader(GL.GL_FRAGMENT_SHADER,"""
    #version 330 core

    in vec3 position;
    in vec3 normal;
    in vec3 c;
    in float visible;
    in vec2 uv;

    out vec4 color;

    uniform vec3 amb;
    uniform vec3 direct;
    uniform vec3 directDir;
    uniform vec3 camT;
    uniform float specAng;
    uniform float specMag;
    uniform float twoSided;
    uniform sampler2D sampler;
    uniform sampler2D normalMap;

    void main(void)
    {

        // Diffuse component
        vec3 n = normalize(normal);
        vec3 l = -normalize(directDir);
        float ref = max(0, dot(n, l)) + twoSided * max(0, dot(-n, l));
        vec3 d = direct * ref;

        // Ambient + diffuse applied to texture
        vec3 tex = texture(sampler, uv).rgb;
        tex = (amb + d) * tex;

        // Specular component
        vec3 refDir = (2 * dot(n, l) * n - l);
        vec3 camDiff = camT - position;
        vec3 H = normalize((l + normalize(camDiff)));
        float refDiff = max(0, dot(refDir, H));
        vec3 shine = direct * specMag * pow(refDiff, specAng);

        color = vec4(visible * (tex + shine), 1.0);
    }
    """)

    fragment_tex_ao_shader=create_and_compile_shader(GL.GL_FRAGMENT_SHADER,"""
    #version 330 core

    in vec3 position;
    in vec3 normal;
    in vec3 c;
    in float visible;
    in vec2 uv;

    out vec4 color;

    uniform vec3 amb;
    uniform vec3 direct;
    uniform vec3 directDir;
    uniform vec3 camT;
    uniform float specAng;
    uniform float specMag;
    uniform float twoSided;
    uniform sampler2D sampler;
    uniform sampler2D normalMap;
    uniform sampler2D aoMap;

    void main(void)
    {

        // Diffuse component
        vec3 n = normalize(normal);
        vec3 l = -normalize(directDir);
        float ref = max(0, dot(n, l)) + twoSided * max(0, dot(-n, l));
        vec3 d = direct * ref;

        // Ambient + diffuse applied to texture
        vec3 tex = texture(sampler, uv).rgb;
        tex = (amb + d) * tex;

        // Specular component
        vec3 refDir = (2 * dot(n, l) * n - l);
        vec3 camDiff = camT - position;
        vec3 H = normalize((l + normalize(camDiff)));
        float refDiff = max(0, dot(refDir, H));
        vec3 shine = direct * specMag * pow(refDiff, specAng);

        vec3 ao = texture(aoMap, uv).rgb;
        color = vec4(visible * (tex * ao + shine), 1.0);
    }
    """)

    fragment_wireframe_shader=create_and_compile_shader(GL.GL_FRAGMENT_SHADER,"""
    #version 330 core

    in vec3 c;

    out vec4 color;

    void main(void)
    {
        color = vec4(c, 1.0);
    }
    """)

    fragment_sh_shader=create_and_compile_shader(GL.GL_FRAGMENT_SHADER,"""
    #version 330 core

    #define C1 0.429043
    #define C2 0.511664
    #define C3 0.743125
    #define C4 0.886227
    #define C5 0.247708

    in vec3 position;
    in vec3 normal;
    in vec3 c;
    in float visible;

    out vec4 color;

    uniform vec3 b0, b1, b2, b3, b4, b5, b6, b7, b8;

    void main(void)
    {
        vec3 n = normalize(normal);
        vec3 r = vec3(0.0f,0.0f,0.0f);
        r += b0 * C4;                            // 1          (L00)
        r += b1 * 2 * C2 * n.y;                  // Y          (L1-1)
        r += b2 * 2 * C2 * n.z;                  // Z          (L10)
        r += b3 * 2 * C2 * n.x;                  // X          (L11)
        r += b4 * 2 * C1 * n.y * n.x;            // YX         (L2-2)
        r += b5 * 2 * C1 * n.y * n.z;            // YZ         (L2-1)
        r += b6 * ((C3 * n.z * n.z) - C5);       // 3Z^2 - 1   (L20)
        r += b7 * 2 * C1 * n.x * n.z;            // XZ         (L21)
        r += b8 * C1 * (n.x * n.x - n.y * n.y);  // X^2 - Y^2  (L22)

        color = vec4(max(c * r, vec3(0.0)), 1.0);
    }
    """)

    fragment_tex_sh_shader=create_and_compile_shader(GL.GL_FRAGMENT_SHADER,"""
    #version 330 core

    #define C1 0.429043
    #define C2 0.511664
    #define C3 0.743125
    #define C4 0.886227
    #define C5 0.247708

    in vec3 position;
    in vec3 normal;
    in vec3 c;
    in float visible;
    in vec2 uv;
    in vec3 tangent;
    in vec3 bitangent;

    out vec4 color;

    uniform vec3 b0, b1, b2, b3, b4, b5, b6, b7, b8;
    uniform sampler2D sampler;
    uniform sampler2D normalMap;

    void main(void)
    {
        vec3 n = normalize(normal);
        vec3 r = vec3(0.0f,0.0f,0.0f);
        r += b0 * C4;                            // 1          (L00)
        r += b1 * 2 * C2 * n.y;                  // Y          (L1-1)
        r += b2 * 2 * C2 * n.z;                  // Z          (L10)
        r += b3 * 2 * C2 * n.x;                  // X          (L11)
        r += b4 * 2 * C1 * n.y * n.x;            // YX         (L2-2)
        r += b5 * 2 * C1 * n.y * n.z;            // YZ         (L2-1)
        r += b6 * ((C3 * n.z * n.z) - C5);       // 3Z^2 - 1   (L20)
        r += b7 * 2 * C1 * n.x * n.z;            // XZ         (L21)
        r += b8 * C1 * (n.x * n.x - n.y * n.y);  // X^2 - Y^2  (L22)

        vec3 cOut = texture(sampler, uv).rgb;
        color = vec4(max(cOut * r, vec3(0.0)), 1.0);
    }
    """)

    fragment_tex_ao_sh_shader=create_and_compile_shader(GL.GL_FRAGMENT_SHADER,"""
    #version 330 core

    #define C1 0.429043
    #define C2 0.511664
    #define C3 0.743125
    #define C4 0.886227
    #define C5 0.247708

    in vec3 position;
    in vec3 normal;
    in vec3 c;
    in float visible;
    in vec2 uv;
    in vec3 tangent;
    in vec3 bitangent;

    out vec4 color;

    uniform vec3 b0, b1, b2, b3, b4, b5, b6, b7, b8;
    uniform sampler2D sampler;
    uniform sampler2D normalMap;
    uniform sampler2D aoMap;

    void main(void)
    {
        vec3 n = normalize(normal);
        vec3 r = vec3(0.0f,0.0f,0.0f);
        r += b0 * C4;                            // 1          (L00)
        r += b1 * 2 * C2 * n.y;                  // Y          (L1-1)
        r += b2 * 2 * C2 * n.z;                  // Z          (L10)
        r += b3 * 2 * C2 * n.x;                  // X          (L11)
        r += b4 * 2 * C1 * n.y * n.x;            // YX         (L2-2)
        r += b5 * 2 * C1 * n.y * n.z;            // YZ         (L2-1)
        r += b6 * ((C3 * n.z * n.z) - C5);       // 3Z^2 - 1   (L20)
        r += b7 * 2 * C1 * n.x * n.z;            // XZ         (L21)
        r += b8 * C1 * (n.x * n.x - n.y * n.y);  // X^2 - Y^2  (L22)

        vec3 cOut = texture(sampler, uv).rgb;
        vec3 ao = texture(aoMap, uv).rgb;
        color = vec4(max(cOut * ao * r, vec3(0.0)), 1.0);
    }
    """)

    vertex_id_shader=create_and_compile_shader(GL.GL_VERTEX_SHADER,"""
    #version 330 core

    layout(location = 0) in vec3 vertexPosition;
    layout(location = 2) in vec3 vertexColor;

    out vec3 c;

    uniform mat4 MVP;

    void main(void)
    {
        gl_Position = MVP * vec4(vertexPosition,1);
        c = vertexColor;
    }
    """)

    fragment_id_shader=create_and_compile_shader(GL.GL_FRAGMENT_SHADER,"""
    #version 330 core

    in vec3 c;

    out vec4 color;

    void main(void)
    {
        color = vec4(c,1.0);
    }
    """)

    vertex_lines_shader=create_and_compile_shader(GL.GL_VERTEX_SHADER,"""
    #version 330 core

    layout(location = 0) in vec3 vertexPosition;
    layout(location = 1) in vec3 lineColor;

    out vec3 c;

    uniform mat4 MVP;
    uniform float bias;

    void main(void)
    {
        gl_Position = MVP * vec4(vertexPosition, 1);
        gl_Position.z += bias;
        c = lineColor;
    }
    """)

    fragment_lines_shader=create_and_compile_shader(GL.GL_FRAGMENT_SHADER,"""
    #version 330 core

    in vec3 c;

    out vec4 color;

    void main(void)
    {
        color = vec4(c, 1.0);
    }
    """)

    id_program = GL.glCreateProgram()
    GL.glAttachShader(id_program, vertex_id_shader)
    GL.glAttachShader(id_program, fragment_id_shader)
    GL.glLinkProgram(id_program)

    program = GL.glCreateProgram()
    GL.glAttachShader(program, vertex_shader)
    GL.glAttachShader(program, fragment_shader)
    GL.glLinkProgram(program)

    program_sh = GL.glCreateProgram()
    GL.glAttachShader(program_sh, vertex_shader)
    GL.glAttachShader(program_sh, fragment_sh_shader)
    GL.glLinkProgram(program_sh)

    program_sh_tex = GL.glCreateProgram()
    GL.glAttachShader(program_sh_tex, vertex_tex_shader)
    GL.glAttachShader(program_sh_tex, fragment_tex_sh_shader)
    GL.glLinkProgram(program_sh_tex)

    program_sh_tex_ao = GL.glCreateProgram()
    GL.glAttachShader(program_sh_tex_ao, vertex_tex_shader)
    GL.glAttachShader(program_sh_tex_ao, fragment_tex_ao_sh_shader)
    GL.glLinkProgram(program_sh_tex_ao)

    program_tex = GL.glCreateProgram()
    GL.glAttachShader(program_tex, vertex_tex_shader)
    GL.glAttachShader(program_tex, fragment_tex_shader)
    GL.glLinkProgram(program_tex)

    program_tex_ao = GL.glCreateProgram()
    GL.glAttachShader(program_tex_ao, vertex_tex_shader)
    GL.glAttachShader(program_tex_ao, fragment_tex_ao_shader)
    GL.glLinkProgram(program_tex_ao)

    program_wireframe = GL.glCreateProgram()
    GL.glAttachShader(program_wireframe, vertex_wireframe_shader)
    GL.glAttachShader(program_wireframe, fragment_wireframe_shader)
    GL.glLinkProgram(program_wireframe)

    program_lines = GL.glCreateProgram()
    GL.glAttachShader(program_lines, vertex_lines_shader)
    GL.glAttachShader(program_lines, fragment_lines_shader)
    GL.glLinkProgram(program_lines)
