`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2^233) Inversion using Itoh-Tsujii algorithm
// Field polynomial: x^233 + x^74 + 1
// Optimal addition chain: 12 multiplications, 232 squarings
//////////////////////////////////////////////////////////////////////////////////
module gf2m_inv233(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [232:0] a,
    output reg  [232:0] inv,
    output reg         done
);

    // --------------------------------------------------------
    // Intermediate registers
    // --------------------------------------------------------
    reg [232:0] b1, b2, b4, b8, b16, b32, b64, b128, b192, b224, b232;

    // --------------------------------------------------------
    // Squarer interface
    // --------------------------------------------------------
    reg  [232:0] sqr_in;
    wire [232:0] sqr_out;
    
    squerer_233 U_SQR (
        .data_in(sqr_in),
        .data_out(sqr_out)
    );

    // --------------------------------------------------------
    // Multiplier interface (3-cycle pipeline assumed)
    // --------------------------------------------------------
    reg  [232:0] mul_a, mul_b;
    wire [232:0] mul_res;

    gf2m_mult233 U_MULT (
        .clk(clk),
        .a(mul_a),
        .b(mul_b),
        .c(mul_res)
    );

    // --------------------------------------------------------
    // FSM states
    // --------------------------------------------------------
    parameter IDLE        = 7'd0,
              STEP2_SQR   = 7'd1,  STEP2_MUL   = 7'd2,  STEP2_WAIT   = 7'd3,
              STEP4_SQR   = 7'd4,  STEP4_MUL   = 7'd5,  STEP4_WAIT   = 7'd6,
              STEP8_SQR   = 7'd7,  STEP8_MUL   = 7'd8,  STEP8_WAIT   = 7'd9,
              STEP16_SQR  = 7'd10, STEP16_MUL  = 7'd11, STEP16_WAIT  = 7'd12,
              STEP32_SQR  = 7'd13, STEP32_MUL  = 7'd14, STEP32_WAIT  = 7'd15,
              STEP64_SQR  = 7'd16, STEP64_MUL  = 7'd17, STEP64_WAIT  = 7'd18,
              STEP128_SQR = 7'd19, STEP128_MUL = 7'd20, STEP128_WAIT = 7'd21,
              STEP192_SQR = 7'd22, STEP192_MUL = 7'd23, STEP192_WAIT = 7'd24,
              STEP224_SQR = 7'd25, STEP224_MUL = 7'd26, STEP224_WAIT = 7'd27,
              STEP232_SQR = 7'd28, STEP232_MUL = 7'd29, STEP232_WAIT = 7'd30,
              STEP233_SQR = 7'd31, DONE        = 7'd32;

    reg [6:0] state;
    reg [8:0] sqr_count;   // up to 233 squarings
    reg [1:0] mul_count;   // 3-cycle multiplier latency

    // --------------------------------------------------------
    // FSM
    // --------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            inv       <= 0;
            done      <= 0;
            sqr_count <= 0;
            mul_count <= 0;
        end else begin
            case (state)

                // -------------------------------
                // IDLE
                // -------------------------------
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        b1    <= a;   // ?1 = a
                        state <= STEP2_SQR;
                    end
                end

                // -------------------------------
                // ?2 = (?1)^2 * ?1
                // -------------------------------
                STEP2_SQR: begin
                    sqr_in <= b1;
                    state  <= STEP2_MUL;
                end
                STEP2_MUL: begin
                    mul_a <= sqr_out; mul_b <= b1;
                    mul_count <= 0;
                    state <= STEP2_WAIT;
                end
                STEP2_WAIT: begin
                    if (mul_count == 3) begin
                        b2 <= mul_res;
                        state <= STEP4_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?4 = (?2)^(2^2) * ?2
                // -------------------------------
                STEP4_SQR: begin
                    if (sqr_count < 2) begin
                        sqr_in <= (sqr_count == 0) ? b2 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b2;
                        mul_count <= 0;
                        state <= STEP4_WAIT;
                    end
                end
                STEP4_WAIT: begin
                    if (mul_count == 3) begin
                        b4 <= mul_res;
                        state <= STEP8_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?8 = (?4)^(2^4) * ?4
                // -------------------------------
                STEP8_SQR: begin
                    if (sqr_count < 4) begin
                        sqr_in <= (sqr_count == 0) ? b4 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b4;
                        mul_count <= 0;
                        state <= STEP8_WAIT;
                    end
                end
                STEP8_WAIT: begin
                    if (mul_count == 3) begin
                        b8 <= mul_res;
                        state <= STEP16_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?16 = (?8)^(2^8) * ?8
                // -------------------------------
                STEP16_SQR: begin
                    if (sqr_count < 8) begin
                        sqr_in <= (sqr_count == 0) ? b8 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b8;
                        mul_count <= 0;
                        state <= STEP16_WAIT;
                    end
                end
                STEP16_WAIT: begin
                    if (mul_count == 3) begin
                        b16 <= mul_res;
                        state <= STEP32_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?32 = (?16)^(2^16) * ?16
                // -------------------------------
                STEP32_SQR: begin
                    if (sqr_count < 16) begin
                        sqr_in <= (sqr_count == 0) ? b16 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b16;
                        mul_count <= 0;
                        state <= STEP32_WAIT;
                    end
                end
                STEP32_WAIT: begin
                    if (mul_count == 3) begin
                        b32 <= mul_res;
                        state <= STEP64_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?64 = (?32)^(2^32) * ?32
                // -------------------------------
                STEP64_SQR: begin
                    if (sqr_count < 32) begin
                        sqr_in <= (sqr_count == 0) ? b32 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b32;
                        mul_count <= 0;
                        state <= STEP64_WAIT;
                    end
                end
                STEP64_WAIT: begin
                    if (mul_count == 3) begin
                        b64 <= mul_res;
                        state <= STEP128_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?128 = (?64)^(2^64) * ?64
                // -------------------------------
                STEP128_SQR: begin
                    if (sqr_count < 64) begin
                        sqr_in <= (sqr_count == 0) ? b64 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b64;
                        mul_count <= 0;
                        state <= STEP128_WAIT;
                    end
                end
                STEP128_WAIT: begin
                    if (mul_count == 3) begin
                        b128 <= mul_res;
                        state <= STEP192_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?192 = (?128)^(2^64) * ?64
                // -------------------------------
                STEP192_SQR: begin
                    if (sqr_count < 64) begin
                        sqr_in <= (sqr_count == 0) ? b128 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b64;
                        mul_count <= 0;
                        state <= STEP192_WAIT;
                    end
                end
                STEP192_WAIT: begin
                    if (mul_count == 3) begin
                        b192 <= mul_res;
                        state <= STEP224_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?224 = (?192)^(2^32) * ?32
                // -------------------------------
                STEP224_SQR: begin
                    if (sqr_count < 32) begin
                        sqr_in <= (sqr_count == 0) ? b192 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b32;
                        mul_count <= 0;
                        state <= STEP224_WAIT;
                    end
                end
                STEP224_WAIT: begin
                    if (mul_count == 3) begin
                        b224 <= mul_res;
                        state <= STEP232_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?232 = (?224)^(2^8) * ?8
                // -------------------------------
                STEP232_SQR: begin
                    if (sqr_count < 8) begin
                        sqr_in <= (sqr_count == 0) ? b224 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b8;
                        mul_count <= 0;
                        state <= STEP232_WAIT;
                    end
                end
                STEP232_WAIT: begin
                    if (mul_count == 3) begin
                        b232 <= mul_res;
                        state <= STEP233_SQR;
                    end else mul_count <= mul_count + 1;
                end
                // -------------------------------
                // ?233 = (?232)^2 * ?1
                // -------------------------------
                STEP233_SQR: begin
                    sqr_in <= b232;
                    state  <= DONE;
                end
                // -------------------------------
                // DONE
                // -------------------------------
                DONE: begin
                    inv  <= sqr_out;
                    done <= 1;
                    if (!start) begin
                        state <= IDLE;
                    end
                end
            endcase
        end
    end

endmodule
