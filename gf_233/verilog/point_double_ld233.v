`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Point Doubling in Lopez-Dahab projective coordinates
// Curve: sect233r1/sect233k1 (y^2 + xy = x^3 + ax^2 + b)
// Example sect233r1: a = 1
// b = 0x066647ede6c332c7f8c0923bb58213b333b20e9ce42
//
// Latencies:
//   - Squarer : 1 cycle
//   - Adder   : 1 cycle (XOR)
//   - Mult    : 3 cycles (FSM waits using mult_cnt)
//////////////////////////////////////////////////////////////////////////////////
module point_double_ld233 (
    input  wire          clk,
    input  wire          rst,
    input  wire          start,
    input  wire [232:0]  X1, Y1, Z1,
    output reg  [232:0]  X2, Y2, Z2,
    output reg           done
);
    // Constants for sect233r1
    localparam [232:0] b_curve = 233'h066647ede6c332c7f8c0923bb58213b333b20e9ce4281fe115f7d8f90ad;
    localparam [232:0] a_curve = 233'd1;

    // Wires for arithmetic units
    reg  [232:0] A, B, C;
    wire [232:0] mult_out, sqr_out;

    // Registers for temporaries
    reg [232:0] X1_sq, Z1_sq, Z1_4, X1_4, Y1_sq;
    reg [232:0] bZ1_4, left;

    // Multiplier counter (3 cycles)
    reg [1:0] mult_cnt;

    // Attach arithmetic units (233-bit versions)
    gf2m_mult233 U_MULT (.clk(clk), .a(A), .b(B), .c(mult_out));
    squerer_233  U_SQR  (.data_in(C), .data_out(sqr_out));

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
