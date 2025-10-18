`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// GF(2^571) Inversion using Itoh-Tsujii Algorithm
// Field polynomial: x^571 + x^10 + x^5 + x^2 + 1
// Addition chain: 15 multiplications, 570 squarings
//////////////////////////////////////////////////////////////////////////////////
module gf2m_inv571(
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    input  wire [570:0] a,
    output reg  [570:0] inv,
    output reg          done
);

    // --------------------------------------------------------
    // Intermediate registers
    // --------------------------------------------------------
    reg [570:0] b1, b2, b4, b8, b16, b32, b64, b128, b256, b512;
    reg [570:0] b544, b560, b568, b570;

    // --------------------------------------------------------
    // Squarer interface
    // --------------------------------------------------------
    reg  [570:0] sqr_in;
    wire [570:0] sqr_out;

    squerer_571 U_SQR (
        .data_in(sqr_in),
        .data_out(sqr_out)
    );

    // --------------------------------------------------------
    // Multiplier interface (3-cycle pipeline assumed)
    // --------------------------------------------------------
    reg  [570:0] mul_a, mul_b;
    wire [570:0] mul_res;

    gf2m_mult571 U_MULT (
        .clk(clk),
        .a(mul_a),
        .b(mul_b),
        .c(mul_res)
    );

    // --------------------------------------------------------
    // FSM states
    // --------------------------------------------------------
    parameter IDLE        = 8'd0,
              STEP2_SQR   = 8'd1,  STEP2_MUL   = 8'd2,  STEP2_WAIT   = 8'd3,
              STEP4_SQR   = 8'd4,  STEP4_MUL   = 8'd5,  STEP4_WAIT   = 8'd6,
              STEP8_SQR   = 8'd7,  STEP8_MUL   = 8'd8,  STEP8_WAIT   = 8'd9,
              STEP16_SQR  = 8'd10, STEP16_MUL  = 8'd11, STEP16_WAIT  = 8'd12,
              STEP32_SQR  = 8'd13, STEP32_MUL  = 8'd14, STEP32_WAIT  = 8'd15,
              STEP64_SQR  = 8'd16, STEP64_MUL  = 8'd17, STEP64_WAIT  = 8'd18,
              STEP128_SQR = 8'd19, STEP128_MUL = 8'd20, STEP128_WAIT = 8'd21,
              STEP256_SQR = 8'd22, STEP256_MUL = 8'd23, STEP256_WAIT = 8'd24,
              STEP512_SQR = 8'd25, STEP512_MUL = 8'd26, STEP512_WAIT = 8'd27,
              STEP544_SQR = 8'd28, STEP544_MUL = 8'd29, STEP544_WAIT = 8'd30,
              STEP560_SQR = 8'd31, STEP560_MUL = 8'd32, STEP560_WAIT = 8'd33,
              STEP568_SQR = 8'd34, STEP568_MUL = 8'd35, STEP568_WAIT = 8'd36,
              STEP570_SQR = 8'd37, STEP570_MUL = 8'd38, STEP570_WAIT = 8'd39,
              STEP571_SQR = 8'd40, DONE        = 8'd41;

    reg [7:0] state;
    reg [9:0] sqr_count;   // up to 571 squarings
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
                        state <= STEP256_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?256 = (?128)^(2^128) * ?128
                // -------------------------------
                STEP256_SQR: begin
                    if (sqr_count < 128) begin
                        sqr_in <= (sqr_count == 0) ? b128 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b128;
                        mul_count <= 0;
                        state <= STEP256_WAIT;
                    end
                end
                STEP256_WAIT: begin
                    if (mul_count == 3) begin
                        b256 <= mul_res;
                        state <= STEP512_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?512 = (?256)^(2^256) * ?256
                // -------------------------------
                STEP512_SQR: begin
                    if (sqr_count < 256) begin
                        sqr_in <= (sqr_count == 0) ? b256 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b256;
                        mul_count <= 0;
                        state <= STEP512_WAIT;
                    end
                end
                STEP512_WAIT: begin
                    if (mul_count == 3) begin
                        b512 <= mul_res;
                        state <= STEP544_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?544 = (?512)^(2^32) * ?32
                // -------------------------------
                STEP544_SQR: begin
                    if (sqr_count < 32) begin
                        sqr_in <= (sqr_count == 0) ? b512 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b32;
                        mul_count <= 0;
                        state <= STEP544_WAIT;
                    end
                end
                STEP544_WAIT: begin
                    if (mul_count == 3) begin
                        b544 <= mul_res;
                        state <= STEP560_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?560 = (?544)^(2^16) * ?16
                // -------------------------------
                STEP560_SQR: begin
                    if (sqr_count < 16) begin
                        sqr_in <= (sqr_count == 0) ? b544 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b16;
                        mul_count <= 0;
                        state <= STEP560_WAIT;
                    end
                end
                STEP560_WAIT: begin
                    if (mul_count == 3) begin
                        b560 <= mul_res;
                        state <= STEP568_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?568 = (?560)^(2^8) * ?8
                // -------------------------------
                STEP568_SQR: begin
                    if (sqr_count < 8) begin
                        sqr_in <= (sqr_count == 0) ? b560 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b8;
                        mul_count <= 0;
                        state <= STEP568_WAIT;
                    end
                end
                STEP568_WAIT: begin
                    if (mul_count == 3) begin
                        b568 <= mul_res;
                        state <= STEP570_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?570 = (?568)^(2^2) * ?2
                // -------------------------------
                STEP570_SQR: begin
                    if (sqr_count < 2) begin
                        sqr_in <= (sqr_count == 0) ? b568 : sqr_out;
                        sqr_count <= sqr_count + 1;
                    end else begin
                        sqr_count <= 0;
                        mul_a <= sqr_out; mul_b <= b2;
                        mul_count <= 0;
                        state <= STEP570_WAIT;
                    end
                end
                STEP570_WAIT: begin
                    if (mul_count == 3) begin
                        b570 <= mul_res;
                        state <= STEP571_SQR;
                    end else mul_count <= mul_count + 1;
                end

                // -------------------------------
                // ?571 = (?570)^2
                // -------------------------------
                STEP571_SQR: begin
                    sqr_in <= b570;
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

