`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Point Addition in Lopez-Dahab projective coordinates
// Curve: sect163r2 (y^2 + xy = x^3 + ax^2 + b)
// a = 1
// b = 0x20A601907B8C953CA1481EB10512F78744A3205FD
//
// Latencies:
//   - Squarer : 1 cycle
//   - Adder   : 1 cycle (XOR)
//   - Mult    : 3 cycles (FSM waits using mult_cnt)
//////////////////////////////////////////////////////////////////////////////////
module point_add_ld163 (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    input  wire [162:0] X0, Y0, Z0,    // projective P
    input  wire [162:0] X1, Y1,        // affine Q (Z=1)
    output reg  [162:0] X2, Y2, Z2,    // P+Q
    output reg          done
);

    // Curve params
    localparam [162:0] a_curve = 163'd1;
    localparam [162:0] b_curve = 163'h20A601907B8C953CA1481EB10512F78744A3205FD;

    // Shared mult/sqr units
    reg  [162:0] A, B, C;
    wire [162:0] mult_out, sqr_out;

    // Instantiate blocks
    gf2m_mult163 U_MULT (.clk(clk), .a(A), .b(B), .c(mult_out));
    squerer_163  U_SQR  (.data_in(C), .data_out(sqr_out));

    // Temporaries
    reg [162:0] T1, T2, Areg, Breg, Creg, Dreg, Ereg, Freg, Greg;

    // Mult counter
    reg [1:0] mult_cnt;

    // FSM states
    localparam IDLE    = 6'd0,
               STEP1   = 6'd1,
               STEP2   = 6'd2,
               STEP3   = 6'd3,
               STEP4   = 6'd4,
               STEP5   = 6'd5,
               STEP6A  = 6'd6,
               STEP6B  = 6'd7,
               STEP7   = 6'd8,
               STEP8   = 6'd9,
               STEP9   = 6'd10,
               STEP10  = 6'd11,
               STEP11  = 6'd12,
               STEP12  = 6'd13,
               STEP13  = 6'd14,
               FINISH  = 6'd31;

    reg [5:0] state;

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
                        // Step 1: compute Z0^2
                        C <= Z0;
                        state <= STEP1;
                    end
                end

                // ---- Step 1: A = Y1*Z0^2 + Y0 ----
                STEP1: begin
                    T1 <= sqr_out;          // Z0^2
                    A <= Y1; B <= sqr_out;  // start Y1*Z0^2
                    mult_cnt <= 0;
                    state <= STEP2;
                end
                STEP2: begin
                    if (mult_cnt == 3) begin
                        Areg <= mult_out ^ Y0;  // A = Y1*Z0^2 + Y0
                        // Step 2: B = X1*Z0 + X0
                        A <= X1; B <= Z0;
                        mult_cnt <= 0;
                        state <= STEP3;
                    end else mult_cnt <= mult_cnt + 1;
                end

                // ---- Step 2: B = X1*Z0 + X0 ----
                STEP3: begin
                    if (mult_cnt == 3) begin
                        Breg <= mult_out ^ X0;
                        // Step 3: C = Z0 * B
                        A <= Z0; B <= mult_out ^ X0;
                        mult_cnt <= 0;
                        state <= STEP4;
                    end else mult_cnt <= mult_cnt + 1;
                end

                // ---- Step 3: C = Z0*B ----
                STEP4: begin
                    if (mult_cnt == 3) begin
                        Creg <= mult_out;
                        // Step 4A: compute B^2
                        C <= Breg;
                        state <= STEP5;
                    end else mult_cnt <= mult_cnt + 1;
                end

                // ---- Step 4: D = B^2 * (C + a*Z0^2) ----
                STEP5: begin
                    T2 <= sqr_out;         // B^2
                    A <= a_curve; B <= T1; // a*Z0^2
                    mult_cnt <= 0;
                    state <= STEP6A;
                end

                STEP6A: begin
                    if (mult_cnt == 3) begin
                        // now mult_out = a*Z0^2
                        A <= T2; B <= (Creg ^ mult_out); // start B^2*(C+a*Z0^2)
                        mult_cnt <= 0;
                        state <= STEP6B;
                    end else mult_cnt <= mult_cnt + 1;
                end

                STEP6B: begin
                    if (mult_cnt == 3) begin
                        Dreg <= mult_out;
                        // Step 5: Z2 = C^2
                        C <= Creg;
                        state <= STEP7;
                    end else mult_cnt <= mult_cnt + 1;
                end

                STEP7: begin
                    Z2 <= sqr_out;
                    // Step 6: E = A*C
                    A <= Areg; B <= Creg;
                    mult_cnt <= 0;
                    state <= STEP8;
                end

                STEP8: begin
                    if (mult_cnt == 3) begin
                        Ereg <= mult_out;
                        // Step 7: X2 = A^2 + D + E
                        C <= Areg;
                        state <= STEP9;
                    end else mult_cnt <= mult_cnt + 1;
                end

                STEP9: begin
                    X2 <= sqr_out ^ Dreg ^ Ereg; // X2
                    // Step 8: F = X2 + X1*Z2
                    A <= X1; B <= Z2;
                    mult_cnt <= 0;
                    state <= STEP10;
                end

                STEP10: begin
                    if (mult_cnt == 3) begin
                        Freg <= X2 ^ mult_out;
                        // Step 9: G = X2 + Y1*Z2
                        A <= Y1; B <= Z2;
                        mult_cnt <= 0;
                        state <= STEP11;
                    end else mult_cnt <= mult_cnt + 1;
                end

                STEP11: begin
                    if (mult_cnt == 3) begin
                        Greg <= X2 ^ mult_out;
                        // Step 10: Y2 = E*F + Z2*G
                        A <= Ereg; B <= Freg;
                        mult_cnt <= 0;
                        state <= STEP12;
                    end else mult_cnt <= mult_cnt + 1;
                end

                STEP12: begin
                    if (mult_cnt == 3) begin
                        T1 <= mult_out; // E*F
                        A <= Z2; B <= Greg;
                        mult_cnt <= 0;
                        state <= STEP13;
                    end else mult_cnt <= mult_cnt + 1;
                end

                STEP13: begin
                    if (mult_cnt == 3) begin
                        Y2 <= T1 ^ mult_out;
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
