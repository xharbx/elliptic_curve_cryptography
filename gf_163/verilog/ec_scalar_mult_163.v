`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Scalar Multiplication in GF(2^163)
// Lopez-Dahab projective coordinates
// Curve: sect163r2
// a = 1
// b = 0x20A601907B8C953CA1481EB10512F78744A3205FD
//////////////////////////////////////////////////////////////////////////////////
module ec_scalar_mult_163 (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    input  wire [162:0] k,          // scalar
    input  wire [162:0] Px, Py,     // base point affine (Z=1)
    output reg  [162:0] X, Y,       // result affine
    output reg          done
);

    // Internal registers
    reg [162:0] Qx, Qy, Qz;   // accumulator (Q)
    reg [162:0] Gx, Gy, Gz;   // base point projective
    reg [162:0] k_reg;
    reg [8:0]   bit_idx;      // counts down from 162 to 0
    reg         started;      // flag if first 1 bit seen

    // FSM states
    localparam IDLE        = 4'd0,
               LOOP        = 4'd1,
               DOUBLE      = 4'd2,
               ADD         = 4'd3,
               INVERSION   = 4'd4,
               AFF_X_MUL   = 4'd5,
               AFF_ZINV_SQ = 4'd6,
               AFF_Y_MUL   = 4'd7,
               FINISH      = 4'd8;

    reg [3:0] state;

    // Wires to submodules
    wire [162:0] Dx, Dy, Dz;
    wire [162:0] Ax, Ay, Az;
    wire         dbl_done, add_done;

    // Inverter
    wire [162:0] Zinv;
    wire         inv_done;

    gf2m_inv163 U_INV (
        .clk(clk), .rst(rst), .start(state==INVERSION),
        .a(Qz), .inv(Zinv), .done(inv_done)
    );

    // Mult/Sqr for affine conversion
    reg  [162:0] A, B, C;
    wire [162:0] mult_out, sqr_out;

    gf2m_mult163 U_MULT (.clk(clk), .a(A), .b(B), .c(mult_out));
    squerer_163  U_SQR  (.data_in(C), .data_out(sqr_out));

    reg [162:0] Zinv_sq;
    reg [1:0]   mult_cnt;

    // Instantiate point double
    point_double_ld163 U_DOUBLE (
        .clk(clk), .rst(rst), .start(state==DOUBLE),
        .X1(Qx), .Y1(Qy), .Z1(Qz),
        .X2(Dx), .Y2(Dy), .Z2(Dz),
        .done(dbl_done)
    );

    // Instantiate point add (Q+P)
    point_add_ld163 U_ADD (
        .clk(clk), .rst(rst), .start(state==ADD),
        .X0(Qx), .Y0(Qy), .Z0(Qz),
        .X1(Gx), .Y1(Gy),
        .X2(Ax), .Y2(Ay), .Z2(Az),
        .done(add_done)
    );

    // FSM
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state    <= IDLE;
            done     <= 0;
            Qx <= 0; Qy <= 0; Qz <= 0;
            Gx <= 0; Gy <= 0; Gz <= 1;
            k_reg <= 0; bit_idx <= 0;
            started <= 0;
            mult_cnt <= 0;
        end else begin
            case (state)

                IDLE: begin
                    done <= 0;
                    if (start) begin
                        Gx <= Px;
                        Gy <= Py;
                        Gz <= 163'd1;
                        k_reg   <= k;
                        bit_idx <= 162;   // start from MSB
                        started <= 0;
                        state   <= LOOP;
                    end
                end

                LOOP: begin
                    if (!started) begin
                        if (k_reg[bit_idx]) begin
                            // First 1 bit ? load Q = P
                            Qx <= Px;
                            Qy <= Py;
                            Qz <= 163'd1;
                            started <= 1;
                            // IMPORTANT: do NOT decrement bit_idx here
                        end else begin
                            // keep skipping zeros
                            if (bit_idx == 0)
                                state <= FINISH;
                            else
                                bit_idx <= bit_idx - 1;
                        end
                    end else begin
                        // Already started ? process normally
                        state <= DOUBLE;
                        if (bit_idx == 0)
                            state <= INVERSION;
                        else
                            bit_idx <= bit_idx - 1;
                    end
                end

                DOUBLE: begin
                    if (dbl_done) begin
                        Qx <= Dx; Qy <= Dy; Qz <= Dz;
                        if (k_reg[bit_idx]) begin
                            state <= ADD;
                        end else begin
                            state <= LOOP;
                        end
                    end
                end

                ADD: begin
                    if (add_done) begin
                        Qx <= Ax; Qy <= Ay; Qz <= Az;
                        state <= LOOP;
                    end
                end 
                
                // Start inversion of Qz
                INVERSION: begin
                    if (inv_done) begin
                        // Start affine X = Qx * Zinv
                        A <= Qx; B <= Zinv;
                        mult_cnt <= 0;
                        state <= AFF_X_MUL;
                    end
                end

                // Multiply Qx*Zinv
                AFF_X_MUL: begin
                    if (mult_cnt == 3) begin
                        X <= mult_out;          // X_aff
                        C <= Zinv;              // prepare square Zinv
                        mult_cnt <= 0;
                        state <= AFF_ZINV_SQ;
                    end else mult_cnt <= mult_cnt + 1;
                end

                // Square Zinv ? Zinv^2
                AFF_ZINV_SQ: begin
                    Zinv_sq <= sqr_out;
                    A <= Qy; B <= sqr_out;
                    mult_cnt <= 0;
                    state <= AFF_Y_MUL;
                end

                // Multiply Qy*Zinv^2
                AFF_Y_MUL: begin
                    if (mult_cnt == 3) begin
                        Y <= mult_out;          // Y_aff
                        done <= 1;
                        state <= FINISH;
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
