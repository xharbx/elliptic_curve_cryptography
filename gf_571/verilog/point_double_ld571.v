`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Point Doubling in Lopez-Dahab projective coordinates
// Curve: sect571r1 (y^2 + xy = x^3 + ax^2 + b)
// a = 1
// b = 0x02F40E7E2221F295DE297117B7F3D62F5C6A97FFCB8CEFF1CD6BA8CE4A9A18AD84FFABBD8EFA593321E45FEB8C39BACA8B9E6E2A9EDD10B141E3A
//
// Latencies:
//   - Squarer : 1 cycle
//   - Adder   : 1 cycle (XOR)
//   - Mult    : 3 cycles (FSM waits using mult_cnt)
//////////////////////////////////////////////////////////////////////////////////
module point_double_ld571 (
    input  wire          clk,
    input  wire          rst,
    input  wire          start,
    input  wire [570:0]  X1, Y1, Z1,
    output reg  [570:0]  X2, Y2, Z2,
    output reg           done
);
    // Constants for sect571r1
    localparam [570:0] b_curve = 571'h02f40e7e2221f295de297117b7f3d62f5c6a97ffcb8ceff1cd6ba8ce4a9a18ad84ffabbd8efa59332be7ad6756a66e294afd185a78ff12aa520e4de739baca0c7ffeff7f2955727a;
    localparam [570:0] a_curve = 571'd1;

    // Wires for arithmetic units
    reg  [570:0] A, B, C;
    wire [570:0] mult_out, sqr_out;

    // Registers for temporaries
    reg [570:0] X1_sq, Z1_sq, Z1_4, X1_4, Y1_sq;
    reg [570:0] bZ1_4, left;

    // Multiplier counter (3 cycles)
    reg [1:0] mult_cnt;

    // Attach arithmetic units (571-bit versions)
    gf2m_mult571 U_MULT (.clk(clk), .a(A), .b(B), .c(mult_out));
    squerer_571  U_SQR  (.data_in(C), .data_out(sqr_out));

    // FSM states
    localparam IDLE       = 5'd0,
               STEP1_X1SQ = 5'd1,
               STEP2_Z1SQ = 5'd2,
               STEP3_Z1_4 = 5'd3,
               STEP4_X1_4 = 5'd4,
               STEP5_Y1SQ = 5'd5,
               STEP6_Z2   = 5'd6,
               STEP7_BZ1  = 5'd7,
               STEP8_LEFT = 5'd8,
               STEP9_AZ2  = 5'd9,
               STEP10_X2R = 5'd10,
               FINISH     = 5'd11;

    reg [4:0] state;

    // FSM
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done  <= 0;
            X2 <= 0; Y2 <= 0; Z2 <= 0;
            mult_cnt <= 0;
        end else begin
            case (state)

                IDLE: begin
                    done <= 0;
                    if (start) begin
                        if (Z1 == 0) begin
                            X2 <= 1; Y2 <= 0; Z2 <= 0; // point at infinity
                            done <= 1;
                            state <= IDLE;
                        end else begin
                            C <= X1; // start squaring X1
                            state <= STEP1_X1SQ;
                        end
                    end
                end

                // ---- 5 squarings ----
                STEP1_X1SQ: begin X1_sq <= sqr_out; C <= Z1;      state <= STEP2_Z1SQ; end
                STEP2_Z1SQ: begin Z1_sq <= sqr_out; C <= sqr_out; state <= STEP3_Z1_4; end
                STEP3_Z1_4: begin Z1_4  <= sqr_out; C <= X1_sq;   state <= STEP4_X1_4; end
                STEP4_X1_4: begin X1_4  <= sqr_out; C <= Y1;      state <= STEP5_Y1SQ; end
                STEP5_Y1SQ: begin 
                    Y1_sq <= sqr_out; 
                    A <= X1_sq; B <= Z1_sq; 
                    mult_cnt <= 0; 
                    state <= STEP6_Z2; 
                end

                // ---- Mult: Z2 = X1^2 * Z1^2 ----
                STEP6_Z2: begin
                    if (mult_cnt == 3) begin
                        Z2 <= mult_out;
                        A <= b_curve; B <= Z1_4;
                        mult_cnt <= 0;
                        state <= STEP7_BZ1;
                    end else mult_cnt <= mult_cnt + 1;
                end

                // ---- Mult: b*Z1^4 ----
                STEP7_BZ1: begin
                    if (mult_cnt == 3) begin
                        bZ1_4 <= mult_out;
                        X2    <= X1_4 ^ mult_out; // X2 = X1^4 + b*Z1^4
                        A <= mult_out; B <= Z2;
                        mult_cnt <= 0;
                        state <= STEP8_LEFT;
                    end else mult_cnt <= mult_cnt + 1;
                end

                // ---- Mult: left = (b*Z1^4)*Z2 ----
                STEP8_LEFT: begin
                    if (mult_cnt == 3) begin
                        left <= mult_out;
                        A <= a_curve; B <= Z2;
                        mult_cnt <= 0;
                        state <= STEP9_AZ2;
                    end else mult_cnt <= mult_cnt + 1;
                end

                // ---- Mult: a*Z2 ----
                STEP9_AZ2: begin
                    if (mult_cnt == 3) begin
                        A <= X2; B <= mult_out ^ Y1_sq ^ bZ1_4;
                        mult_cnt <= 0;
                        state <= STEP10_X2R;
                    end else mult_cnt <= mult_cnt + 1;
                end

                // ---- Mult: X2 * (...) ----
                STEP10_X2R: begin
                    if (mult_cnt == 3) begin
                        Y2 <= left ^ mult_out;
                        state <= FINISH;
                        done <= 1;
                    end else mult_cnt <= mult_cnt + 1;
                end

                FINISH: begin
                    done <= 0;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
